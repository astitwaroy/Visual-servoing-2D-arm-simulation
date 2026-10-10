import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arm_math import L1, L2, forward_kinematics, inverse_kinematics_2link  # noqa: E402
from camera import PinholeCamera  # noqa: E402
from kalman import ConstantVelocityKF  # noqa: E402
from perception import calibrate_camera, detect_target, pixel_to_world  # noqa: E402
from servo_pipeline import SimConfig, run_episode  # noqa: E402


# ------------------------------------------------------------------ kinematics
@pytest.mark.parametrize("elbow_up", [True, False])
def test_ik_fk_roundtrip(elbow_up):
    rng = np.random.default_rng(0)
    for _ in range(500):
        r = rng.uniform(abs(L1 - L2) + 1e-3, L1 + L2 - 1e-3)
        a = rng.uniform(-math.pi, math.pi)
        x, y = r * math.cos(a), r * math.sin(a)
        sol = inverse_kinematics_2link(x, y, elbow_up=elbow_up)
        assert sol is not None
        _, (ex, ey) = forward_kinematics(*sol)
        assert math.hypot(ex - x, ey - y) < 1e-9


def test_ik_unreachable():
    assert inverse_kinematics_2link(L1 + L2 + 0.01, 0) is None
    assert inverse_kinematics_2link(0.05, 0) is None


# ---------------------------------------------------------------------- camera
def test_projection_roundtrip_and_workspace_in_view():
    cam = PinholeCamera()
    for a in np.linspace(0, 2 * math.pi, 37):
        p = ((L1 + L2) * math.cos(a), (L1 + L2) * math.sin(a))
        u, v = cam.world_to_pixel(p)
        assert 0 <= u < cam.width and 0 <= v < cam.height, "workspace must be fully visible"
        back = cam.pixel_to_world((u, v))
        assert math.hypot(back[0] - p[0], back[1] - p[1]) < 1e-9


def test_calibration_recovers_homography():
    cam = PinholeCamera()
    H, mean_err = calibrate_camera(cam, rng=np.random.default_rng(1))
    assert mean_err < 0.01  # under 1 cm on the calibration dots
    rng = np.random.default_rng(2)
    for _ in range(50):
        p = rng.uniform(-2.0, 2.0, 2)
        u, v = cam.world_to_pixel(p)
        q = pixel_to_world(H, u, v)
        assert math.hypot(q[0] - p[0], q[1] - p[1]) < 0.02


# ------------------------------------------------------------------ perception
def test_detection_accuracy_with_noise_and_clutter():
    cam = PinholeCamera()
    H, _ = calibrate_camera(cam, rng=np.random.default_rng(3))
    rng = np.random.default_rng(4)
    errors = []
    for _ in range(100):
        r, a = rng.uniform(0.4, 2.1), rng.uniform(0, 2 * math.pi)
        target = (r * math.cos(a), r * math.sin(a))
        frame = cam.render((rng.uniform(-3, 3), rng.uniform(-2, 2)), target, rng=rng,
                           pixel_noise_std=8, brightness_jitter=0.08,
                           distractors=[tuple(rng.uniform(-2, 2, 2)) for _ in range(4)])
        det = detect_target(frame)
        assert det is not None
        w = pixel_to_world(H, det.u, det.v)
        errors.append(math.hypot(w[0] - target[0], w[1] - target[1]))
    assert np.mean(errors) < 0.01
    assert np.max(errors) < 0.03


def test_no_target_means_no_detection():
    cam = PinholeCamera()
    rng = np.random.default_rng(5)
    frame = cam.render((0.3, 0.6), None, rng=rng, pixel_noise_std=8,
                       distractors=[tuple(rng.uniform(-2, 2, 2)) for _ in range(6)])
    assert detect_target(frame) is None


# ---------------------------------------------------------------------- kalman
def test_kalman_beats_raw_measurements_and_estimates_velocity():
    rng = np.random.default_rng(6)
    kf = ConstantVelocityKF(accel_noise=0.05, meas_std=0.03)
    dt, v = 0.05, np.array([0.4, -0.2])
    raw_err, kf_err, vel = [], [], []
    for k in range(400):
        truth = np.array([0.5, 0.5]) + v * k * dt
        z = truth + rng.normal(0, 0.03, 2)
        kf.predict(dt)
        kf.update(tuple(z))
        if k > 50:
            raw_err.append(np.linalg.norm(z - truth))
            kf_err.append(np.linalg.norm(np.array(kf.position()) - truth))
            vel.append(kf.velocity())
    assert np.mean(kf_err) < 0.5 * np.mean(raw_err)
    # instantaneous velocity is noisy; its average must match the true velocity
    assert np.linalg.norm(np.mean(vel, axis=0) - v) < 0.02


def test_kalman_gate_rejects_outlier():
    kf = ConstantVelocityKF(meas_std=0.02)
    for k in range(40):
        kf.predict(0.05)
        kf.update((1.0 + 0.01 * k, 0.0))
    assert kf.update((-1.5, 1.5)) is False  # a jump of ~3 m in one frame is not the ball


# ------------------------------------------------------------------ closed loop
def test_closed_loop_kalman_beats_raw_under_latency():
    base = dict(steps=300, target="circular", seed=0)
    _, raw, _ = run_episode(SimConfig(estimator="raw", **base))
    _, kal, _ = run_episode(SimConfig(estimator="kalman", **base))
    assert kal["rms_error_m"] < 0.6 * raw["rms_error_m"]
    assert kal["rms_error_m"] < 0.05
    assert kal["detection_pct"] > 80
