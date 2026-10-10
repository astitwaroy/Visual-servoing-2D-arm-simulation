import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from arm_math import forward_kinematics, inverse_kinematics_2link
from visual_servo_controller import cartesian_step_controller


# Simulation parameters
DT = 0.05
STEPS = 800

# Target trajectory parameters
TARGET_RADIUS = 1.6
TARGET_ANGULAR_SPEED = 0.3  # rad/s


def target_position(t: float):
    """
    Moving target: a circle in front of the robot.
    """
    angle = TARGET_ANGULAR_SPEED * t
    x = TARGET_RADIUS * math.cos(angle)
    y = TARGET_RADIUS * math.sin(angle)
    return x, y


# Initial joint angles
theta1 = 0.0
theta2 = 0.0


# Matplotlib setup
fig, ax = plt.subplots()
ax.set_aspect("equal", "box")
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_title("2-Link Arm – Visual Servoing (Tracking Moving Target)")

# Plots
base_plot, = ax.plot([], [], "ko", markersize=6)
elbow_plot, = ax.plot([], [], "bo", markersize=6)
end_plot, = ax.plot([], [], "ro", markersize=6)
link1_line, = ax.plot([], [], "b-", linewidth=3)
link2_line, = ax.plot([], [], "r-", linewidth=3)

target_plot, = ax.plot([], [], "gx", markersize=8, label="Target")
path_plot, = ax.plot([], [], "g--", linewidth=1, alpha=0.4, label="Target path")

# For drawing the full circular trajectory once
circle_x = []
circle_y = []
for k in range(200):
    tt = k * 2 * math.pi / 200.0
    cx = TARGET_RADIUS * math.cos(tt)
    cy = TARGET_RADIUS * math.sin(tt)
    circle_x.append(cx)
    circle_y.append(cy)
path_plot.set_data(circle_x, circle_y)

ax.legend(loc="upper left")


def init():
    base_plot.set_data([0], [0])
    return base_plot, elbow_plot, end_plot, link1_line, link2_line, target_plot, path_plot


def update(frame_idx):
    global theta1, theta2

    t = frame_idx * DT

    # 1) Current FK
    (x1, y1), (x2, y2) = forward_kinematics(theta1, theta2)
    p_ee = (x2, y2)

    # 2) Target position at time t
    xt, yt = target_position(t)
    p_target = (xt, yt)

    # 3) Visual-servo-like controller: small step towards target in Cartesian space
    p_next = cartesian_step_controller(p_ee, p_target, gain=0.8, max_step=0.08)
    x_next, y_next = p_next

    # 4) IK to get new joint angles for p_next
    ik = inverse_kinematics_2link(x_next, y_next, elbow_up=True)
    if ik is not None:
        theta1, theta2 = ik
        # Update FK after applying new angles
        (x1, y1), (x2, y2) = forward_kinematics(theta1, theta2)
        p_ee = (x2, y2)
    # else: if IK fails, keep old theta and positions

    # 5) Update plots
    base_plot.set_data([0], [0])
    elbow_plot.set_data([x1], [y1])
    end_plot.set_data([p_ee[0]], [p_ee[1]])

    link1_line.set_data([0, x1], [0, y1])
    link2_line.set_data([x1, p_ee[0]], [y1, p_ee[1]])

    target_plot.set_data([p_target[0]], [p_target[1]])

    return base_plot, elbow_plot, end_plot, link1_line, link2_line, target_plot, path_plot


if __name__ == "__main__":
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=STEPS,
        interval=50,
        blit=True,
    )

    # Toggle this later if you want a GIF:
    SAVE_GIF = True
    if SAVE_GIF:
        import os
        from matplotlib.animation import PillowWriter
        os.makedirs("assets", exist_ok=True)

        print("⚠ Saving GIF... this may take 5–20 sec depending on your PC")

        anim._interval = 60     # fewer frames
        anim.save_count = 200   # limit total frames written

        gif_writer = PillowWriter(fps=15)
        anim.save("assets/sim_visual_servo_moving_circular_target.gif", writer=gif_writer, dpi=70)
        print("Saved GIF to assets/sim_visual_servo_moving_circular_target.gif")
    else:
        plt.show()
