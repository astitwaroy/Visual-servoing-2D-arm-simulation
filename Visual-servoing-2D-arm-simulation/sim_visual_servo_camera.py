"""
Camera-guided visual servoing with real perception in the loop.

Left panel:  world view - arm, true target, what the robot believes (estimate)
             and the raw camera measurements.
Right panel: the (delayed) camera frame the robot actually received, with
             the detection drawn on top.

Examples
--------
python sim_visual_servo_camera.py                         # live window, Kalman, random target
python sim_visual_servo_camera.py --estimator raw         # no filter: watch it lag and jitter
python sim_visual_servo_camera.py --target circular --latency 5
python sim_visual_servo_camera.py --save-gif assets/sim_visual_servo_camera_kalman.gif
python sim_visual_servo_camera.py --benchmark             # oracle vs raw vs kalman table
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from servo_pipeline import ESTIMATORS, CameraServoLoop, SimConfig, run_episode, summarize


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--estimator", choices=ESTIMATORS, default="kalman")
    p.add_argument("--target", choices=("random", "circular"), default="random")
    p.add_argument("--speed", type=float, default=None, help="max speed m/s (random) or rad/s (circular)")
    p.add_argument("--latency", type=int, default=3, help="camera latency in 50 ms frames (default 3 = 150 ms)")
    p.add_argument("--noise", type=float, default=8.0, help="sensor noise std-dev in grey levels")
    p.add_argument("--shake", type=float, default=1.5, help="camera vibration std-dev in pixels")
    p.add_argument("--drop", type=float, default=0.05, help="probability a frame is lost")
    p.add_argument("--no-occlusion", action="store_true", help="disable the sweeping occluder")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--save-gif", metavar="PATH", help="render a GIF instead of opening a window")
    p.add_argument("--gif-frames", type=int, default=240)
    p.add_argument("--benchmark", action="store_true", help="compare estimators over several seeds and exit")
    p.add_argument("--seeds", type=int, default=5, help="number of seeds for --benchmark")
    return p.parse_args(argv)


def config_from_args(a) -> SimConfig:
    return SimConfig(
        estimator=a.estimator, target=a.target, target_speed=a.speed, latency_frames=a.latency,
        pixel_noise_std=a.noise, shake_px=a.shake, drop_prob=a.drop,
        occlusions=not a.no_occlusion, seed=a.seed, steps=a.steps,
    )


# ---------------------------------------------------------------------- benchmark
def benchmark(base: SimConfig, n_seeds: int) -> None:
    print(f"Target: {base.target} | latency {base.latency_frames * base.dt * 1000:.0f} ms | "
          f"noise {base.pixel_noise_std} | shake {base.shake_px} px | drop {base.drop_prob:.0%} | "
          f"{n_seeds} seeds x {base.steps} steps\n")
    header = f"{'estimator':<10}{'RMS err':>10}{'p95 err':>10}{'max err':>10}{'locked<5cm':>12}{'detected':>10}"
    print(header)
    print("-" * len(header))
    results = {}
    for est in ESTIMATORS:
        ms = []
        for s in range(n_seeds):
            cfg = SimConfig(**{**base.__dict__, "estimator": est, "seed": s})
            _, m, _ = run_episode(cfg)
            ms.append(m)
        avg = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        results[est] = avg
        det = "n/a" if est == "oracle" else f"{avg['detection_pct']:.0f}%"
        print(f"{est:<10}{avg['rms_error_m'] * 100:>8.1f}cm{avg['p95_error_m'] * 100:>8.1f}cm"
              f"{avg['max_error_m'] * 100:>8.1f}cm{avg['locked_pct']:>11.0f}%{det:>10}")
    cut = 100 * (1 - results["kalman"]["rms_error_m"] / results["raw"]["rms_error_m"])
    print(f"\nKalman filter reduces RMS tracking error by {cut:.0f}% versus raw detections.")


# -------------------------------------------------------------------- animation
def animate(cfg: SimConfig, save_gif: str | None, gif_frames: int) -> None:
    import matplotlib
    if save_gif:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle

    from arm_math import L1, L2

    loop = CameraServoLoop(cfg)
    fig, (axw, axc) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.subplots_adjust(left=0.04, right=0.99, top=0.86, bottom=0.06, wspace=0.08)

    # World view
    axw.set_aspect("equal")
    axw.set_xlim(-2.6, 2.6)
    axw.set_ylim(-2.6, 2.6)
    axw.set_title("World frame (m)", fontsize=10)
    axw.add_patch(Circle((0, 0), L1 + L2, fill=False, ls="--", lw=0.8, color="0.6"))
    axw.grid(alpha=0.25)
    link1, = axw.plot([], [], "-", lw=5, color="#245ca0", solid_capstyle="round")
    link2, = axw.plot([], [], "-", lw=4, color="#e0661c", solid_capstyle="round")
    joints, = axw.plot([], [], "o", ms=6, color="k")
    meas_pts, = axw.plot([], [], ".", ms=3, color="0.45", alpha=0.6, label="camera measurements")
    true_trail, = axw.plot([], [], "-", lw=1, color="#1f8a4c", alpha=0.35)
    true_pt, = axw.plot([], [], "x", ms=10, mew=2.5, color="#1f8a4c", label="true target")
    est_pt, = axw.plot([], [], "o", ms=11, mfc="none", mew=2, color="#b8187a", label=f"estimate ({cfg.estimator})")
    axw.legend(loc="lower left", fontsize=8, framealpha=0.85)

    # Camera view
    axc.set_xticks([])
    axc.set_yticks([])
    img_artist = axc.imshow(np.zeros((loop.camera.height, loop.camera.width, 3), np.uint8))
    det_circle = Circle((0, 0), 1, fill=False, lw=2, color="#ff00aa")
    axc.add_patch(det_circle)
    cam_title = axc.set_title("", fontsize=10)

    title = fig.suptitle("", fontsize=11)
    meas_hist, true_hist = [], []

    def update(_):
        log = loop.step()
        (x1, y1), (x2, y2) = log.elbow, log.ee
        link1.set_data([0, x1], [0, y1])
        link2.set_data([x1, x2], [y1, y2])
        joints.set_data([0, x1, x2], [0, y1, y2])
        true_hist.append(log.target)
        true_hist[:] = true_hist[-200:]
        true_trail.set_data(*zip(*true_hist))
        true_pt.set_data([log.target[0]], [log.target[1]])
        if log.measurement is not None:
            meas_hist.append(log.measurement)
            meas_hist[:] = meas_hist[-60:]
        if meas_hist:
            meas_pts.set_data(*zip(*meas_hist))
        if log.estimate is not None:
            est_pt.set_data([log.estimate[0]], [log.estimate[1]])

        if loop.last_frame is not None:
            img_artist.set_data(loop.last_frame[:, :, ::-1])  # BGR -> RGB
            d = loop.last_detection
            det_circle.set_visible(d is not None)
            if d is not None:
                det_circle.center = (d.u, d.v)
                det_circle.radius = d.radius + 4
            lag_ms = (log.t - loop.last_frame_time) * 1000
            status = "target detected" if d is not None else "NO DETECTION (dropped / occluded)"
            cam_title.set_text(f"Camera frame from {lag_ms:.0f} ms ago  |  {status}")
        else:
            cam_title.set_text("Camera: waiting for first frame")

        recent = [l for l in loop.logs[-100:] if l.t >= cfg.warmup_s]
        rms_txt = ""
        if recent:
            rms = math.sqrt(sum(l.error ** 2 for l in recent) / len(recent))
            rms_txt = f"  (RMS last {len(recent) * cfg.dt:.0f} s: {rms * 100:.1f} cm)"
        title.set_text(f"Visual servoing with camera in the loop  |  estimator: {cfg.estimator}  |  "
                       f"t = {log.t:5.1f} s  |  tracking error {log.error * 100:4.1f} cm{rms_txt}")
        return link1, link2, joints, meas_pts, true_trail, true_pt, est_pt, img_artist, det_circle, cam_title, title

    if save_gif:
        os.makedirs(os.path.dirname(save_gif) or ".", exist_ok=True)
        # Skip the first second so the GIF opens mid-track
        for _ in range(20):
            loop.step()
        if shutil.which("ffmpeg"):
            # Better colours and much smaller files than matplotlib's PillowWriter:
            # draw every simulation step, keep every other one (10 fps = real time),
            # and let ffmpeg build one global palette.
            with tempfile.TemporaryDirectory() as tmp:
                for i in range(gif_frames):
                    update(i)
                    if i % 2 == 0:
                        fig.savefig(os.path.join(tmp, f"f{i // 2:05d}.png"), dpi=72)
                vf = ("scale=680:-1:flags=lanczos,hqdn3d=6:0:6:0,split[a][b];"
                      "[a]palettegen=max_colors=128:stats_mode=full[p];"
                      "[b][p]paletteuse=dither=none:diff_mode=rectangle")
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "10",
                                "-i", os.path.join(tmp, "f%05d.png"), "-vf", vf, save_gif], check=True)
        else:
            anim = FuncAnimation(fig, update, frames=gif_frames, interval=50, blit=False)
            anim.save(save_gif, writer=PillowWriter(fps=20), dpi=72)
        print(f"Saved {save_gif}")
        print({k: round(v, 4) for k, v in summarize(loop.logs, cfg).items()})
    else:
        anim = FuncAnimation(fig, update, frames=cfg.steps, interval=50, blit=False, repeat=False)  # noqa: F841
        plt.show()


def main(argv=None) -> int:
    a = parse_args(argv)
    cfg = config_from_args(a)
    if a.benchmark:
        cfg.steps = min(cfg.steps, 600)
        benchmark(cfg, a.seeds)
        return 0
    animate(cfg, a.save_gif, a.gif_frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
