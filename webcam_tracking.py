"""
Live demo: move a coloured ball in front of your webcam and the simulated arm follows it.

The same perception -> Kalman -> controller -> IK chain as the simulation, but the
camera is real. The image is mapped onto the arm's workspace (centre of the image =
arm base, image width = 4.8 m), so moving the ball around the frame moves the target
around the arm.

Usage
-----
python webcam_tracking.py                      # default webcam, green ball
python webcam_tracking.py --color red          # red / blue / yellow / green
python webcam_tracking.py --video clip.mp4     # run on a recorded video instead
python webcam_tracking.py --video clip.mp4 --headless --out result.mp4

Keys (window mode): q / Esc quit · k toggle Kalman filter · m toggle mask view
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from arm_math import L1, L2, forward_kinematics, inverse_kinematics_2link
from kalman import ConstantVelocityKF
from perception import COLOR_RANGES, color_mask, detect_target
from visual_servo_controller import cartesian_step_controller

WORLD_HALF_WIDTH = 2.4    # metres shown across half the image width
PANEL = 480               # size of the arm panel in pixels
NOMINAL_DT = 0.05         # controller limits are tuned for 20 Hz


def image_to_world(u: float, v: float, w: int, h: int) -> Tuple[float, float]:
    s = WORLD_HALF_WIDTH / (w / 2)
    return (u - w / 2) * s, -(v - h / 2) * s


def world_to_panel(p: Tuple[float, float]) -> Tuple[int, int]:
    s = PANEL / (2 * WORLD_HALF_WIDTH)
    return int(round(PANEL / 2 + p[0] * s)), int(round(PANEL / 2 - p[1] * s))


def draw_panel(theta, target: Optional[Tuple[float, float]], estimate: Optional[Tuple[float, float]],
               use_kf: bool, error: Optional[float], fps: float) -> np.ndarray:
    img = np.full((PANEL, PANEL, 3), 246, np.uint8)
    s = PANEL / (2 * WORLD_HALF_WIDTH)
    for g in np.arange(-2, 2.01, 1.0):
        a = world_to_panel((g, 0))[0]
        b = world_to_panel((0, g))[1]
        cv2.line(img, (a, 0), (a, PANEL), (225, 225, 225), 1)
        cv2.line(img, (0, b), (PANEL, b), (225, 225, 225), 1)
    cv2.circle(img, world_to_panel((0, 0)), int((L1 + L2) * s), (200, 200, 200), 1, cv2.LINE_AA)

    (x1, y1), (x2, y2) = forward_kinematics(*theta)
    p0, p1, p2 = world_to_panel((0, 0)), world_to_panel((x1, y1)), world_to_panel((x2, y2))
    cv2.line(img, p0, p1, (160, 92, 36), 9, cv2.LINE_AA)
    cv2.line(img, p1, p2, (28, 102, 224), 7, cv2.LINE_AA)
    for p, r in ((p0, 7), (p1, 5), (p2, 5)):
        cv2.circle(img, p, r, (30, 30, 30), -1, cv2.LINE_AA)

    if target is not None:
        cv2.drawMarker(img, world_to_panel(target), (76, 138, 31), cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)
    if estimate is not None:
        cv2.circle(img, world_to_panel(estimate), 11, (122, 24, 184), 2, cv2.LINE_AA)

    lines = [f"filter: {'Kalman' if use_kf else 'raw'}  (k)", f"{fps:4.1f} fps"]
    if error is not None:
        lines.insert(1, f"tip-to-ball: {error * 100:5.1f} cm")
    for i, t in enumerate(lines):
        cv2.putText(img, t, (10, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    return img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--video", help="read from a video file instead of the webcam")
    ap.add_argument("--color", choices=sorted(COLOR_RANGES), default="green")
    ap.add_argument("--no-kalman", action="store_true")
    ap.add_argument("--mirror", action="store_true", help="flip the webcam image horizontally (selfie view)")
    ap.add_argument("--min-area", type=float, default=150.0, help="smallest blob area in pixels")
    ap.add_argument("--headless", action="store_true", help="no window (use with --out)")
    ap.add_argument("--out", help="write the side-by-side view to this video file")
    a = ap.parse_args(argv)

    cap = cv2.VideoCapture(a.video if a.video else a.camera)
    if not cap.isOpened():
        print("Could not open the camera/video. Try --camera 1 or --video path.", file=sys.stderr)
        return 1
    file_fps = cap.get(cv2.CAP_PROP_FPS) if a.video else 0.0

    kf = ConstantVelocityKF(accel_noise=2.0, meas_std=0.03)  # a hand moves more abruptly than the sim target
    use_kf = not a.no_kalman
    show_mask = False
    theta = (math.pi / 2, -math.pi / 2)
    writer = None
    t_prev = time.perf_counter()
    fps = 0.0
    frames = detections = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if a.mirror:
            frame = cv2.flip(frame, 1)
        frames += 1
        h, w = frame.shape[:2]

        now = time.perf_counter()
        dt = (1.0 / file_fps) if file_fps > 0 else max(1e-3, now - t_prev)
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-3, now - t_prev)) if fps else 1.0 / max(1e-3, now - t_prev)
        t_prev = now

        det = detect_target(frame, color=a.color, min_area=a.min_area, min_circularity=0.5)
        meas = image_to_world(det.u, det.v, w, h) if det else None
        detections += det is not None

        kf.predict(dt)
        if meas is not None:
            kf.update(meas)
        estimate = (kf.position() if kf.initialized else None) if use_kf else meas

        if estimate is not None:
            (_, _), ee = forward_kinematics(*theta)
            scale = dt / NOMINAL_DT
            p_next = cartesian_step_controller(ee, estimate, gain=min(1.0, 0.8 * scale), max_step=0.08 * scale)
            sol = inverse_kinematics_2link(*p_next, elbow_up=True)
            if sol is not None:
                theta = sol
        (_, _), ee = forward_kinematics(*theta)
        err = math.hypot(ee[0] - meas[0], ee[1] - meas[1]) if meas else None

        # Left: camera view with detection overlay
        view = cv2.cvtColor(color_mask(frame, a.color), cv2.COLOR_GRAY2BGR) if show_mask else frame.copy()
        if det:
            cv2.circle(view, (int(det.u), int(det.v)), int(det.radius) + 4, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(view, (int(det.u), int(det.v)), (255, 0, 255), cv2.MARKER_CROSS, 12, 2)
        else:
            cv2.putText(view, f"no {a.color} ball found", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        view = cv2.resize(view, (int(w * PANEL / h), PANEL))
        panel = draw_panel(theta, meas, estimate, use_kf, err, fps if not file_fps else file_fps)
        combo = np.hstack([view, panel])

        if a.out:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(a.out, fourcc, file_fps or 20.0, (combo.shape[1], combo.shape[0]))
            writer.write(combo)

        if not a.headless:
            cv2.imshow("Visual servoing - webcam", combo)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("k"):
                use_kf = not use_kf
            if key == ord("m"):
                show_mask = not show_mask

    cap.release()
    if writer is not None:
        writer.release()
    if not a.headless:
        cv2.destroyAllWindows()
    print(f"Processed {frames} frames, ball detected in {detections} ({100 * detections / max(1, frames):.0f}%).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
