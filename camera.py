"""
Synthetic pinhole camera looking down at the arm's working plane (z = 0).

The controller no longer receives the target position directly. Instead this
module renders what an overhead camera would see (a 640x480 BGR image), and
the perception module has to find the target in that image.

Conventions
-----------
- World frame: x right, y up, z out of the table, origin at the arm base.
- Camera frame (OpenCV): x right, y down, z forward (into the scene).
- Extrinsics: X_cam = R @ (X_world - C), where C is the camera centre in world.
- Because every point of interest lies on the plane z = 0, the mapping between
  world (x, y) and pixels (u, v) is a 3x3 homography H = K [r1 r2 t].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from arm_math import L1, L2, forward_kinematics

# Target appearance (BGR) and physical size
TARGET_RADIUS_M = 0.09
TARGET_BGR = (60, 200, 70)        # green ball
ARM_LINK1_BGR = (160, 90, 40)     # blue-ish link
ARM_LINK2_BGR = (40, 110, 220)    # orange link
GRIPPER_BGR = (85, 85, 85)
BASE_BGR = (95, 95, 95)


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


@dataclass
class PinholeCamera:
    width: int = 640
    height: int = 480
    fx: float = 400.0
    fy: float = 400.0
    cx: float = 320.0
    cy: float = 240.0
    # Camera centre in world coordinates (metres)
    position: Tuple[float, float, float] = (0.1, 0.32, 4.5)
    # Small mounting errors so the image is not a trivial scaled copy of the world
    tilt_deg: float = 4.0
    yaw_deg: float = 3.0

    K: np.ndarray = field(init=False, repr=False)
    R: np.ndarray = field(init=False, repr=False)
    t: np.ndarray = field(init=False, repr=False)
    H: np.ndarray = field(init=False, repr=False)      # world plane -> pixel
    H_inv: np.ndarray = field(init=False, repr=False)  # pixel -> world plane

    def __post_init__(self) -> None:
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=float)
        # Looking straight down: cam x = world x, cam y = -world y, cam z = -world z
        r_down = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
        self.R = _rot_z(math.radians(self.yaw_deg)) @ _rot_x(math.radians(self.tilt_deg)) @ r_down
        C = np.asarray(self.position, dtype=float)
        self.t = -self.R @ C
        self.H = self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], self.t])
        self.H /= self.H[2, 2]
        self.H_inv = np.linalg.inv(self.H)

    # ------------------------------------------------------------------ geometry
    def world_to_pixel(self, xy: Sequence[float]) -> Tuple[float, float]:
        return apply_homography(self.H, xy)

    def pixel_to_world(self, uv: Sequence[float]) -> Tuple[float, float]:
        return apply_homography(self.H_inv, uv)

    def metres_to_pixels(self, xy: Sequence[float], length_m: float) -> float:
        """Approximate image length of a short world segment at point xy."""
        u0, v0 = self.world_to_pixel(xy)
        u1, v1 = self.world_to_pixel((xy[0] + length_m, xy[1]))
        return math.hypot(u1 - u0, v1 - v0)

    # ----------------------------------------------------------------- rendering
    def render(
        self,
        theta: Tuple[float, float],
        target_xy: Optional[Sequence[float]],
        rng: Optional[np.random.Generator] = None,
        pixel_noise_std: float = 0.0,
        brightness_jitter: float = 0.0,
        distractors: Iterable[Sequence[float]] = (),
        markers: Iterable[Sequence[float]] = (),
        occluders: Iterable[np.ndarray] = (),
        shake_px: float = 0.0,
    ) -> np.ndarray:
        """
        Draw one camera frame (BGR uint8).

        theta              joint angles of the arm
        target_xy          world position of the green target (None = not visible)
        pixel_noise_std    std-dev of additive Gaussian sensor noise (grey levels)
        brightness_jitter  +/- fraction of global brightness change per frame
        distractors        world points where small green-ish clutter appears
        markers            world points drawn as black calibration dots
        occluders          pixel-space polygons drawn over everything (e.g. a hand reaching in)
        shake_px           std-dev of random image translation, i.e. camera-mount vibration
        """
        rng = rng if rng is not None else np.random.default_rng()
        img = self._background()

        for m in markers:
            u, v = self.world_to_pixel(m)
            r = max(3, int(round(self.metres_to_pixels(m, 0.05))))
            cv2.circle(img, (int(round(u)), int(round(v))), r, (20, 20, 20), -1, cv2.LINE_AA)

        for d in distractors:
            u, v = self.world_to_pixel(d)
            r = max(2, int(round(self.metres_to_pixels(d, 0.03))))
            cv2.circle(img, (int(round(u)), int(round(v))), r, (80, 170, 90), -1, cv2.LINE_AA)

        self._draw_arm(img, theta)

        if target_xy is not None:
            # Ball sits above the arm plane, so it is drawn on top of the arm.
            u, v = self.world_to_pixel(target_xy)
            r = self.metres_to_pixels(target_xy, TARGET_RADIUS_M)
            shift = 4  # sub-pixel drawing precision (1/16 px)
            cv2.circle(
                img,
                (int(round(u * (1 << shift))), int(round(v * (1 << shift)))),
                int(round(r * (1 << shift))),
                TARGET_BGR, -1, cv2.LINE_AA, shift,
            )

        for poly in occluders:
            cv2.fillPoly(img, [np.round(poly).astype(np.int32)], (120, 125, 130), cv2.LINE_AA)

        if shake_px > 0:
            dx, dy = rng.normal(0.0, shake_px, 2)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            img = cv2.warpAffine(img, M, (self.width, self.height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)

        if brightness_jitter > 0:
            gain = 1.0 + rng.uniform(-brightness_jitter, brightness_jitter)
            img = cv2.convertScaleAbs(img, alpha=gain)  # saturating scale, stays uint8
        if pixel_noise_std > 0:
            # OpenCV's RNG is much faster than numpy for full frames; seed it from `rng`
            # so runs stay reproducible.
            cv2.setRNGSeed(int(rng.integers(0, 2**31 - 1)))
            noise = np.empty(img.shape, np.int16)
            cv2.randn(noise, 0.0, float(pixel_noise_std))
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img

    def _background(self) -> np.ndarray:
        if not hasattr(self, "_bg_cache"):
            h, w = self.height, self.width
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            # soft vignette, like an overhead lamp
            d = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / math.hypot(w / 2, h / 2)
            base = 205 - 45 * d ** 2
            bg = np.stack([base * 0.97, base * 0.98, base], axis=-1)
            # workspace outline on the table (reach limit)
            bg = bg.astype(np.uint8)
            pts = []
            for k in range(181):
                a = 2 * math.pi * k / 180
                pts.append(self.world_to_pixel(((L1 + L2) * math.cos(a), (L1 + L2) * math.sin(a))))
            cv2.polylines(bg, [np.round(np.array(pts)).astype(np.int32)], True, (170, 170, 175), 1, cv2.LINE_AA)
            self._bg_cache = bg
        return self._bg_cache.copy()

    def _draw_arm(self, img: np.ndarray, theta: Tuple[float, float]) -> None:
        (x1, y1), (x2, y2) = forward_kinematics(theta[0], theta[1])
        p0 = self.world_to_pixel((0.0, 0.0))
        p1 = self.world_to_pixel((x1, y1))
        p2 = self.world_to_pixel((x2, y2))
        ip = lambda p: (int(round(p[0])), int(round(p[1])))
        w1 = max(4, int(round(self.metres_to_pixels((0, 0), 0.09))))
        w2 = max(3, int(round(self.metres_to_pixels((0, 0), 0.07))))
        cv2.line(img, ip(p0), ip(p1), ARM_LINK1_BGR, w1, cv2.LINE_AA)
        cv2.line(img, ip(p1), ip(p2), ARM_LINK2_BGR, w2, cv2.LINE_AA)
        cv2.circle(img, ip(p0), w1, BASE_BGR, -1, cv2.LINE_AA)
        cv2.circle(img, ip(p1), max(3, w1 // 2 + 1), (230, 230, 230), -1, cv2.LINE_AA)
        cv2.circle(img, ip(p2), max(3, w2 // 2 + 1), GRIPPER_BGR, -1, cv2.LINE_AA)


def apply_homography(H: np.ndarray, p: Sequence[float]) -> Tuple[float, float]:
    vec = H @ np.array([p[0], p[1], 1.0])
    return float(vec[0] / vec[2]), float(vec[1] / vec[2])
