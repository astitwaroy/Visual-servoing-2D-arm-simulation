import math
from typing import Optional, Tuple

# Link lengths (same as previous repos to keep consistency)
L1 = 1.2
L2 = 1.0


def forward_kinematics(theta1: float, theta2: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    2-link planar arm FK.
    Returns:
        (x1, y1) -> elbow position
        (x2, y2) -> end-effector position
    """
    x1 = L1 * math.cos(theta1)
    y1 = L1 * math.sin(theta1)

    x2 = x1 + L2 * math.cos(theta1 + theta2)
    y2 = y1 + L2 * math.sin(theta1 + theta2)

    return (x1, y1), (x2, y2)


def inverse_kinematics_2link(
    x: float,
    y: float,
    elbow_up: bool = True
) -> Optional[Tuple[float, float]]:
    """
    Analytic IK for 2-link planar arm.
    Returns (theta1, theta2) in radians, or None if unreachable.
    """
    r2 = x * x + y * y
    r = math.sqrt(r2)

    # Reachability check
    if r > (L1 + L2) or r < abs(L1 - L2):
        return None

    # Law of cosines for theta2
    cos_t2 = (r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    cos_t2 = max(-1.0, min(1.0, cos_t2))  # numeric safety
    theta2 = math.acos(cos_t2)
    if not elbow_up:
        theta2 = -theta2

    # theta1
    phi = math.atan2(y, x)
    k1 = L1 + L2 * math.cos(theta2)
    k2 = L2 * math.sin(theta2)
    psi = math.atan2(k2, k1)
    theta1 = phi - psi

    return theta1, theta2
