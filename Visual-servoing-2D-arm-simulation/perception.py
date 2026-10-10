"""
Perception: find the target in a camera frame and map it to the world plane.

Pipeline
--------
BGR frame -> HSV -> colour threshold -> morphological clean-up -> contours
-> pick the most plausible blob -> sub-pixel centroid (image moments)
-> homography -> world (x, y)

Also contains a plane calibration routine that estimates the pixel->world
homography from a calibration image with dots at known world positions,
the same way you would calibrate a real overhead camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# HSV ranges in OpenCV units (H: 0-179, S/V: 0-255). Red wraps around 0, so it needs two ranges.
COLOR_RANGES: Dict[str, List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = {
    "green": [((40, 90, 60), (85, 255, 255))],
    "blue": [((95, 120, 60), (130, 255, 255))],
    "yellow": [((20, 110, 110), (35, 255, 255))],
    "red": [((0, 130, 80), (8, 255, 255)), ((170, 130, 80), (179, 255, 255))],
}


@dataclass
class Detection:
    u: float          # centroid column (px)
    v: float          # centroid row (px)
    area: float       # blob area (px^2)
    radius: float     # radius of the minimum enclosing circle (px)
    circularity: float


def color_mask(frame_bgr: np.ndarray, color: str = "green") -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in COLOR_RANGES[color]:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_target(
    frame_bgr: np.ndarray,
    color: str = "green",
    min_area: float = 40.0,
    min_circularity: float = 0.6,
) -> Optional[Detection]:
    """
    Return the most likely target blob, or None if nothing ball-shaped is visible.

    Blobs are scored by area x circularity, so small clutter and elongated
    shapes lose to the ball even when they share its colour.
    """
    mask = color_mask(frame_bgr, color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    best: Optional[Detection] = None
    best_score = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        perim = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (perim * perim) if perim > 0 else 0.0
        if circ < min_circularity:
            continue
        m = cv2.moments(c)
        if m["m00"] <= 0:
            continue
        (_, _), r = cv2.minEnclosingCircle(c)
        score = area * circ
        if score > best_score:
            best_score = score
            best = Detection(m["m10"] / m["m00"], m["m01"] / m["m00"], area, r, circ)
    return best


def pixel_to_world(H_pix_to_world: np.ndarray, u: float, v: float) -> Tuple[float, float]:
    p = H_pix_to_world @ np.array([u, v, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


# ---------------------------------------------------------------------- calibration
def calibration_grid(nx: int = 5, ny: int = 4, spacing: float = 0.8) -> np.ndarray:
    """World positions of calibration dots, row-major from top-left (max y) to bottom-right."""
    xs = (np.arange(nx) - (nx - 1) / 2) * spacing
    ys = (np.arange(ny) - (ny - 1) / 2) * spacing
    pts = [(x, y) for y in ys[::-1] for x in xs]
    return np.array(pts, dtype=np.float64)


def detect_calibration_dots(frame_bgr: np.ndarray, nx: int, ny: int) -> Optional[np.ndarray]:
    """Find nx*ny dark dots and return their pixel centroids in row-major order."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cents = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 15:
            continue
        perim = cv2.arcLength(c, True)
        if perim == 0 or 4 * np.pi * area / perim ** 2 < 0.7:
            continue
        m = cv2.moments(c)
        cents.append((m["m10"] / m["m00"], m["m01"] / m["m00"], area))
    if len(cents) < nx * ny:
        return None
    # keep the nx*ny largest round dark blobs
    cents.sort(key=lambda c: -c[2])
    cents = np.array([c[:2] for c in cents[: nx * ny]])
    # order: rows by v (top to bottom), then columns by u (left to right)
    cents = cents[np.argsort(cents[:, 1])]
    rows = [cents[i * nx:(i + 1) * nx] for i in range(ny)]
    rows = [r[np.argsort(r[:, 0])] for r in rows]
    return np.vstack(rows)


def calibrate_plane(frame_bgr: np.ndarray, world_pts: np.ndarray, nx: int, ny: int) -> Tuple[np.ndarray, float]:
    """
    Estimate the pixel -> world homography from one calibration image.

    Returns (H_pix_to_world, mean reprojection error in metres).
    """
    pix = detect_calibration_dots(frame_bgr, nx, ny)
    if pix is None:
        raise RuntimeError("Calibration failed: could not find all calibration dots")
    H, _ = cv2.findHomography(pix, world_pts, method=0)
    proj = cv2.perspectiveTransform(pix.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = float(np.mean(np.linalg.norm(proj - world_pts, axis=1)))
    return H, err


def calibrate_camera(camera, rng: Optional[np.random.Generator] = None,
                     nx: int = 5, ny: int = 4, spacing: float = 0.8,
                     pixel_noise_std: float = 3.0) -> Tuple[np.ndarray, float]:
    """Render a calibration frame with `camera` (arm folded out of the way) and calibrate from it."""
    world = calibration_grid(nx, ny, spacing)
    # Park the arm along the diagonal, between the dots, so it does not cover them
    frame = camera.render((-3 * np.pi / 4, 0.0), None, rng=rng, pixel_noise_std=pixel_noise_std, markers=world)
    return calibrate_plane(frame, world, nx, ny)
