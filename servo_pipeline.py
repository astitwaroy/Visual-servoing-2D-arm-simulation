"""
Camera-in-the-loop visual servoing.

Each control cycle (dt = 0.05 s):

  1. The target moves (unknown to the robot).
  2. The overhead camera renders a frame: sensor noise, lighting flicker,
     mount vibration, small green clutter, occasional occlusion, dropped frames.
  3. The frame reaches the robot `latency_frames` cycles late.
  4. Perception finds the ball (HSV threshold + contours + moments).
  5. The calibrated homography maps the pixel centroid to world metres.
  6. The estimator turns measurements into a target estimate:
       oracle  - true position, no camera (the original simulation)
       raw     - latest detection, held when nothing is detected
       kalman  - constant-velocity Kalman filter, gated, latency-compensated
  7. The Cartesian step controller + analytic IK move the arm.

`run_episode` returns per-step logs and summary metrics so the three
estimators can be compared on identical target paths and camera noise.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from arm_math import forward_kinematics, inverse_kinematics_2link
from camera import PinholeCamera
from kalman import ConstantVelocityKF
from perception import Detection, calibrate_camera, detect_target, pixel_to_world
from target_motion import CircularTarget, RandomWalkTarget
from visual_servo_controller import cartesian_step_controller

ESTIMATORS = ("oracle", "raw", "kalman")


@dataclass
class SimConfig:
    dt: float = 0.05
    steps: int = 600
    target: str = "random"             # "random" | "circular"
    target_speed: Optional[float] = None  # max m/s (random) or rad/s (circular); None = defaults
    seed: int = 0
    estimator: str = "kalman"          # "oracle" | "raw" | "kalman"

    # Camera realism
    latency_frames: int = 3            # 3 x 50 ms = 150 ms from exposure to controller
    pixel_noise_std: float = 8.0       # grey levels
    brightness_jitter: float = 0.08    # +/- 8 % flicker
    shake_px: float = 1.5              # mount vibration (px, std-dev)
    drop_prob: float = 0.05            # frames lost in transport / motion blur
    n_distractors: int = 4             # small green clutter on the table
    occlusions: bool = True            # a bar sweeps across the view now and then

    # Controller (same defaults as the original sim)
    gain: float = 0.8
    max_step: float = 0.08
    elbow_up: bool = True

    # Kalman tuning
    kf_accel_noise: float = 0.05
    kf_meas_std: float = 0.02

    warmup_s: float = 2.0              # ignored when computing metrics
    lock_threshold: float = 0.05       # metres


@dataclass
class StepLog:
    t: float
    target: Tuple[float, float]
    estimate: Optional[Tuple[float, float]]
    measurement: Optional[Tuple[float, float]]
    ee: Tuple[float, float]
    elbow: Tuple[float, float]
    theta: Tuple[float, float]
    error: float
    detected: bool
    gated_out: bool


class CameraServoLoop:
    def __init__(self, cfg: SimConfig):
        if cfg.estimator not in ESTIMATORS:
            raise ValueError(f"estimator must be one of {ESTIMATORS}")
        self.cfg = cfg
        self.rng_cam = np.random.default_rng(cfg.seed + 1000)
        self.rng_drop = np.random.default_rng(cfg.seed + 2000)

        if cfg.target == "circular":
            self.target = CircularTarget(angular_speed=cfg.target_speed or 0.3)
        else:
            self.target = RandomWalkTarget(max_speed=cfg.target_speed or 0.8, seed=cfg.seed)
        self.target.reset()

        self.camera = PinholeCamera()
        # Calibrate once, from an image, like a real setup. The true homography is never used.
        self.H_pix2world, self.calib_error = calibrate_camera(self.camera, rng=np.random.default_rng(cfg.seed))

        crng = np.random.default_rng(cfg.seed + 3000)
        self.distractors = [tuple(crng.uniform(-2.0, 2.0, 2)) for _ in range(cfg.n_distractors)]

        self.kf = ConstantVelocityKF(accel_noise=cfg.kf_accel_noise, meas_std=cfg.kf_meas_std)
        self.frames: Deque[Tuple[float, np.ndarray]] = deque()
        self.theta = (0.0, 0.0)
        self.t = 0.0
        self.last_raw: Optional[Tuple[float, float]] = None
        self.logs: List[StepLog] = []

        # For visualisation
        self.last_frame: Optional[np.ndarray] = None
        self.last_frame_time: Optional[float] = None
        self.last_detection: Optional[Detection] = None

    # ------------------------------------------------------------------ helpers
    def occluder_polygons(self, t: float) -> List[np.ndarray]:
        """A grey bar (think: a person's arm) sweeps across the image for 2.5 s every 12 s."""
        if not self.cfg.occlusions:
            return []
        period, start, dur = 12.0, 7.0, 2.5
        phase = t % period
        if not (start <= phase < start + dur):
            return []
        w, h = self.camera.width, self.camera.height
        s = (phase - start) / dur
        x = -80 + s * (w + 160)
        bar = 70
        return [np.array([[x, -10], [x + bar, -10], [x + bar + 40, h + 10], [x + 40, h + 10]], dtype=float)]

    # --------------------------------------------------------------------- loop
    def step(self) -> StepLog:
        cfg = self.cfg
        self.t += cfg.dt
        target = self.target.step(cfg.dt)

        measurement = None
        detected = False
        gated_out = False

        if cfg.estimator == "oracle":
            estimate: Optional[Tuple[float, float]] = target
        else:
            frame = self.camera.render(
                self.theta, target, rng=self.rng_cam,
                pixel_noise_std=cfg.pixel_noise_std,
                brightness_jitter=cfg.brightness_jitter,
                distractors=self.distractors,
                occluders=self.occluder_polygons(self.t),
                shake_px=cfg.shake_px,
            )
            self.frames.append((self.t, frame))

            det = None
            if len(self.frames) > cfg.latency_frames:
                t_frame, delayed = self.frames.popleft()
                self.last_frame, self.last_frame_time = delayed, t_frame
                dropped = self.rng_drop.random() < cfg.drop_prob
                det = None if dropped else detect_target(delayed)
                self.last_detection = det
            if det is not None:
                measurement = pixel_to_world(self.H_pix2world, det.u, det.v)
                detected = True

            latency_s = cfg.latency_frames * cfg.dt
            if cfg.estimator == "raw":
                if measurement is not None:
                    self.last_raw = measurement
                estimate = self.last_raw
            else:
                # Filter time = time of the newest frame; predict a step, fuse if we have a detection.
                self.kf.predict(cfg.dt)
                if measurement is not None:
                    gated_out = not self.kf.update(measurement)
                estimate = self.kf.predict_ahead(latency_s) if self.kf.initialized else None

        (_, _), ee = forward_kinematics(*self.theta)
        if estimate is not None:
            p_next = cartesian_step_controller(ee, estimate, gain=cfg.gain, max_step=cfg.max_step)
            sol = inverse_kinematics_2link(p_next[0], p_next[1], elbow_up=cfg.elbow_up)
            if sol is not None:
                self.theta = sol
        elbow, ee = forward_kinematics(*self.theta)

        log = StepLog(
            t=self.t, target=target, estimate=estimate, measurement=measurement,
            ee=ee, elbow=elbow, theta=self.theta,
            error=math.hypot(ee[0] - target[0], ee[1] - target[1]),
            detected=detected, gated_out=gated_out,
        )
        self.logs.append(log)
        return log

    def run(self, steps: Optional[int] = None) -> List[StepLog]:
        for _ in range(steps if steps is not None else self.cfg.steps):
            self.step()
        return self.logs


def summarize(logs: List[StepLog], cfg: SimConfig) -> Dict[str, float]:
    kept = [l for l in logs if l.t >= cfg.warmup_s]
    err = np.array([l.error for l in kept])
    est_err = np.array([
        math.hypot(l.estimate[0] - l.target[0], l.estimate[1] - l.target[1])
        for l in kept if l.estimate is not None
    ])
    return {
        "rms_error_m": float(np.sqrt(np.mean(err ** 2))),
        "mean_error_m": float(np.mean(err)),
        "p95_error_m": float(np.percentile(err, 95)),
        "max_error_m": float(np.max(err)),
        "locked_pct": float(100 * np.mean(err < cfg.lock_threshold)),
        "estimate_rms_m": float(np.sqrt(np.mean(est_err ** 2))) if est_err.size else float("nan"),
        "detection_pct": float(100 * np.mean([l.detected for l in kept])),
    }


def run_episode(cfg: SimConfig) -> Tuple[List[StepLog], Dict[str, float], CameraServoLoop]:
    loop = CameraServoLoop(cfg)
    logs = loop.run()
    return logs, summarize(logs, cfg), loop
