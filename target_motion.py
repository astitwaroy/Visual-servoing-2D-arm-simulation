"""
Target motion generators.

CircularTarget     known periodic path (used to sanity-check tracking)
RandomWalkTarget   smooth random wandering with no fixed shape; the robot can
                   only follow it through feedback
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


class CircularTarget:
    def __init__(self, radius: float = 1.6, angular_speed: float = 0.3):
        self.radius = radius
        self.angular_speed = angular_speed
        self.t = 0.0

    def reset(self) -> Tuple[float, float]:
        self.t = 0.0
        return self.position()

    def position(self) -> Tuple[float, float]:
        a = self.angular_speed * self.t
        return self.radius * math.cos(a), self.radius * math.sin(a)

    def step(self, dt: float) -> Tuple[float, float]:
        self.t += dt
        return self.position()


class RandomWalkTarget:
    """Random acceleration, clamped speed, softly kept inside a radius (same model as the original sim)."""

    def __init__(self, max_radius: float = 1.8, max_speed: float = 0.8, accel: float = 0.8,
                 start: Tuple[float, float] = (1.2, 0.5), seed: Optional[int] = None):
        self.max_radius = max_radius
        self.max_speed = max_speed
        self.accel = accel
        self.start = start
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> Tuple[float, float]:
        self.x, self.y = self.start
        self.vx = self.vy = 0.0
        return self.x, self.y

    def position(self) -> Tuple[float, float]:
        return self.x, self.y

    def step(self, dt: float) -> Tuple[float, float]:
        self.vx += self.rng.uniform(-self.accel, self.accel) * dt
        self.vy += self.rng.uniform(-self.accel, self.accel) * dt

        speed = math.hypot(self.vx, self.vy)
        if speed > self.max_speed:
            self.vx *= self.max_speed / speed
            self.vy *= self.max_speed / speed

        self.x += self.vx * dt
        self.y += self.vy * dt

        r = math.hypot(self.x, self.y)
        if r > self.max_radius:
            self.vx -= (self.x / r) * self.accel * dt
            self.vy -= (self.y / r) * self.accel * dt
            self.x *= 0.98
            self.y *= 0.98
        return self.x, self.y
