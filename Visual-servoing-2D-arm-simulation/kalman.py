"""
Constant-velocity Kalman filter for a target moving on the plane.

State     x = [px, py, vx, vy]
Model     position integrates velocity; velocity changes by white-noise acceleration
Measure   z = [px, py] from the camera (already mapped to world metres)

Why the robot needs it
----------------------
- Detections jitter (sensor noise, camera shake): the filter smooths them.
- Frames get dropped or the target is occluded: the filter keeps predicting.
- Camera frames arrive late (latency): the filter's velocity estimate lets the
  controller aim where the target is *now*, not where it was when the frame was taken.
- A wrong detection (clutter) is rejected by a chi-square gate on the innovation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

CHI2_2DOF_999 = 13.82  # 99.9 % gate for a 2-D measurement


class ConstantVelocityKF:
    def __init__(self, accel_noise: float = 0.05, meas_std: float = 0.02,
                 init_pos_std: float = 0.5, init_vel_std: float = 1.0,
                 gate: Optional[float] = CHI2_2DOF_999, max_rejects: int = 5):
        self.q = accel_noise          # acceleration noise spectral density (m^2/s^3)
        self.r = meas_std ** 2        # measurement variance (m^2)
        self.init_pos_var = init_pos_std ** 2
        self.init_vel_var = init_vel_std ** 2
        self.gate = gate
        self.max_rejects = max_rejects
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.rejects = 0
        self.last_nis = 0.0

    @property
    def initialized(self) -> bool:
        return self.x is not None

    def init(self, z: Tuple[float, float]) -> None:
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([self.init_pos_var, self.init_pos_var, self.init_vel_var, self.init_vel_var])
        self.rejects = 0

    @staticmethod
    def _F(dt: float) -> np.ndarray:
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        return F

    def _Q(self, dt: float) -> np.ndarray:
        q = self.q
        a, b, c = dt ** 3 / 3, dt ** 2 / 2, dt
        return q * np.array([[a, 0, b, 0], [0, a, 0, b], [b, 0, c, 0], [0, b, 0, c]])

    def predict(self, dt: float) -> None:
        if not self.initialized:
            return
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

    def update(self, z: Tuple[float, float]) -> bool:
        """Fuse a measurement. Returns False if the measurement was gated out as an outlier."""
        if not self.initialized:
            self.init(z)
            return True
        Hm = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        y = np.asarray(z, dtype=float) - Hm @ self.x
        S = Hm @ self.P @ Hm.T + self.r * np.eye(2)
        nis = float(y @ np.linalg.solve(S, y))
        self.last_nis = nis
        if self.gate is not None and nis > self.gate:
            self.rejects += 1
            if self.rejects >= self.max_rejects:
                # Persistent disagreement: the track is lost, so restart from the measurement.
                self.init(z)
                return True
            return False
        self.rejects = 0
        K = self.P @ Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ Hm) @ self.P
        return True

    def position(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    def velocity(self) -> Tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    def predict_ahead(self, horizon: float) -> Tuple[float, float]:
        """Where the target will be `horizon` seconds after the filter's current time (no state change)."""
        return float(self.x[0] + self.x[2] * horizon), float(self.x[1] + self.x[3] * horizon)
