import math
from typing import Tuple


def cartesian_step_controller(
    p_ee: Tuple[float, float],
    p_target: Tuple[float, float],
    gain: float = 0.8,
    max_step: float = 0.08
) -> Tuple[float, float]:
    """
    Simple Cartesian visual servo-like controller.

    Given:
      - p_ee: current end-effector position (x, y)
      - p_target: desired target position (x, y)

    Compute:
      - a small step towards the target:
        p_next = p_ee + step

    Where:
      step = clamp( gain * (p_target - p_ee), norm <= max_step )

    This is basically:
      v = K * error
      p_next = p + v * dt
    with dt absorbed into 'gain' and 'max_step'.
    """
    x, y = p_ee
    xt, yt = p_target

    ex = xt - x
    ey = yt - y

    # Proportional control in Cartesian space
    step_x = gain * ex
    step_y = gain * ey

    step_norm = math.hypot(step_x, step_y)
    if step_norm > max_step and step_norm > 1e-9:
        scale = max_step / step_norm
        step_x *= scale
        step_y *= scale

    return x + step_x, y + step_y
