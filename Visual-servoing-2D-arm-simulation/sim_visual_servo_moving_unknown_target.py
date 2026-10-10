import math
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from arm_math import forward_kinematics, inverse_kinematics_2link
from visual_servo_controller import cartesian_step_controller

# ============================================
# Simulation parameters
# ============================================

DT = 0.05
STEPS = 800

# Target wandering parameters
TARGET_MAX_RADIUS = 1.8     # keep target roughly in a circle of this radius
TARGET_MAX_SPEED = 0.8      # limit on target speed (m/s)
TARGET_ACCEL = 0.8          # random acceleration magnitude

# ============================================
# Target state (global)
# ============================================

# Start at some random point inside the reachable region
target_x = 1.2
target_y = 0.5
target_vx = 0.0
target_vy = 0.0

def update_target(dt: float):
    """
    Smooth random wandering target motion.
    No fixed path shape, no circle, no line.
    Just random-but-smooth drift inside a radius.
    """
    global target_x, target_y, target_vx, target_vy

    # 1) Random acceleration
    ax = random.uniform(-TARGET_ACCEL, TARGET_ACCEL)
    ay = random.uniform(-TARGET_ACCEL, TARGET_ACCEL)

    target_vx += ax * dt
    target_vy += ay * dt

    # 2) Clamp speed
    speed = math.hypot(target_vx, target_vy)
    if speed > TARGET_MAX_SPEED and speed > 1e-9:
        scale = TARGET_MAX_SPEED / speed
        target_vx *= scale
        target_vy *= scale

    # 3) Update position
    target_x += target_vx * dt
    target_y += target_vy * dt

    # 4) Keep target inside a soft radius
    r = math.hypot(target_x, target_y)
    if r > TARGET_MAX_RADIUS:
        # Push it softly back towards the origin
        # by flipping velocity a bit and pulling position inwards
        # (not a hard bounce, more like a soft steering)
        target_vx -= (target_x / r) * TARGET_ACCEL * dt
        target_vy -= (target_y / r) * TARGET_ACCEL * dt

        # small correction on position too, to avoid drifting out
        target_x *= 0.98
        target_y *= 0.98

    return target_x, target_y


# ============================================
# Arm initial state
# ============================================

theta1 = 0.0
theta2 = 0.0

# ============================================
# Matplotlib setup
# ============================================

fig, ax = plt.subplots()
ax.set_aspect("equal", "box")
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_title("2-Link Arm – Visual Servoing (Unknown Moving Target)")

# Arm graphics
base_plot, = ax.plot([], [], "ko", markersize=6)
elbow_plot, = ax.plot([], [], "bo", markersize=6)
end_plot, = ax.plot([], [], "ro", markersize=6)
link1_line, = ax.plot([], [], "b-", linewidth=3)
link2_line, = ax.plot([], [], "r-", linewidth=3)

# Target visualization
target_plot, = ax.plot([], [], "gx", markersize=8, label="Target")
trail_plot, = ax.plot([], [], "g--", linewidth=1, alpha=0.4, label="Target trail")

# To store target positions for trail
trail_x = []
trail_y = []

ax.legend(loc="upper left")


def init():
    base_plot.set_data([0], [0])
    return base_plot, elbow_plot, end_plot, link1_line, link2_line, target_plot, trail_plot


def update(frame_idx):
    global theta1, theta2, trail_x, trail_y

    t = frame_idx * DT

    # 1) Update target with random wandering
    xt, yt = update_target(DT)
    p_target = (xt, yt)

    # 2) Current FK
    (x1, y1), (x2, y2) = forward_kinematics(theta1, theta2)
    p_ee = (x2, y2)

    # 3) Visual-servo-like controller: move end-effector slightly towards target
    p_next = cartesian_step_controller(
        p_ee,
        p_target,
        gain=0.8,
        max_step=0.08
    )
    x_next, y_next = p_next

    # 4) IK for the new desired EE position
    ik = inverse_kinematics_2link(x_next, y_next, elbow_up=True)
    if ik is not None:
        theta1, theta2 = ik
        # Recompute FK using new joint angles
        (x1, y1), (x2, y2) = forward_kinematics(theta1, theta2)
        p_ee = (x2, y2)
    # else: if IK fails, keep old theta and FK result

    # 5) Update arm plots
    base_plot.set_data([0], [0])
    elbow_plot.set_data([x1], [y1])
    end_plot.set_data([p_ee[0]], [p_ee[1]])

    link1_line.set_data([0, x1], [0, y1])
    link2_line.set_data([x1, p_ee[0]], [y1, p_ee[1]])

    # 6) Update target plot + trail
    target_plot.set_data([p_target[0]], [p_target[1]])

    trail_x.append(p_target[0])
    trail_y.append(p_target[1])

    # keep trail length reasonable
    MAX_TRAIL_POINTS = 400
    if len(trail_x) > MAX_TRAIL_POINTS:
        trail_x = trail_x[-MAX_TRAIL_POINTS:]
        trail_y = trail_y[-MAX_TRAIL_POINTS:]

    trail_plot.set_data(trail_x, trail_y)

    return base_plot, elbow_plot, end_plot, link1_line, link2_line, target_plot, trail_plot


if __name__ == "__main__":
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=STEPS,
        interval=50,
        blit=True,
    )

    SAVE_GIF = True  # set True when you want to export
    if SAVE_GIF:
        import os
        from matplotlib.animation import PillowWriter
        os.makedirs("assets", exist_ok=True)

        print("⚠ Saving GIF... this may take 5–20 sec depending on your PC")

        anim._interval = 60     # fewer frames
        anim.save_count = 200   # limit total frames written

        gif_writer = PillowWriter(fps=15)
        anim.save("assets/sim_visual_servo_unknown_target.gif", writer=gif_writer, dpi=70)
        print("Saved GIF to assets/sim_visual_servo_unknown_target.gif")
    else:
        plt.show()
