# 🎯 Visual Servoing — Camera-Guided Manipulation (2D Simulation)

This project implements a **visual servoing control loop**:  
a robot arm uses **camera perception → IK → motion control** to continuously track and reach a moving target.

Unlike previous repos where the target followed a known pattern (circle, animation path),  
here **the target moves unpredictably with an unknown, non-repeating trajectory**, forcing the robot to rely purely on visual feedback — just like real robots do.

---

## 🚀 What is Visual Servoing?

Visual Servoing = **using camera observations to control robot motion.**

| Step | Meaning |
|------|---------|
| Camera sees object | `(x_target_cam, y_target_cam)` |
| Convert to world / map | using extrinsic transform |
| Inverse Kinematics | compute required joint angles |
| Trajectory control | move smoothly toward target |
| Loop | because the target keeps moving |

Instead of planning once, the arm **continuously corrects itself** based on new camera input.

---

## 🔍 What this simulation demonstrates

| Concept | Demonstrated in this project |
|--------|------------------------------|
| Visual feedback loop | Robot corrects motion based on camera perception |
| Real-time IK tracking | Target moves → IK recomputed every frame |
| Unpredictable motion | No fixed pattern — target path constantly changes |
| Jacobian-free servoing | Control happens in joint space directly |
| Smooth motion | No jerks; continuous interpolation |

This is how real robots do:
- Object tracking for pick & place  
- Human handover  
- Dynamic grasping  
- Eye-in-hand manipulation  

---

## 🧠 Architecture Overview

Camera Observation → Target Pose (camera frame)  
↓  
Frame Transform (camera → world)  
↓  
Inverse Kinematics (θ1, θ2)  
↓  
Cubic / proportional joint update  
↓  
Robot Arm Motion  
↓  
Camera sees new position → loop (closed-loop servoing)

The **feedback loop** is the key.

---

## 🦾 Robot Arm Model

A 2-link planar manipulator:

- (base)── L1 ──(joint1)── L2 ──(joint2)── (end effector)

| Symbol | Meaning |
|--------|--------|
| L1, L2 | link lengths |
| θ1     | shoulder joint angle |
| θ2     | elbow joint angle |

Uses closed-form IK, **elbow-up configuration**.

---

## 🔄 Target Motion

The target does **not** follow a circle, ellipse, Lissajous curve, etc.  
Instead, the movement is **random + continuous + unpredictable**, so the robot **must dynamically track** rather than memorize.

Every frame:
- target changes by a random vector
- velocity + direction keep shifting
→ simulates **real object motion with noise + uncertainty**

---

## 📂 Project Structure

visual-servoing-2d/

│

├── assets/

│   ├── sim_visual_servo_moving_circular_trajectory.gif

│   ├── sim_visual_servo_moving_unknown_trajectory.gif

│

├── arm_math.py                       # Forward & inverse kinematics for 2-link arm

├── target_motion.py                  # Unpredictable motion generator for moving target

├── visual_servo_controller.py        # Perception → IK → joint correction feedback loop

│

├── sim_visual_servo_moving_circular_trajectory.py   # Camera-guided tracking of circular path

├── sim_visual_servo_unknown_trajectory.py    # Camera-guided tracking of unknown/random path

│

└── README.md


---

## ▶ Running the Simulation

```bash
python sim_visual_servo_moving_circular_trajectory.py
```
```bash
python sim_visual_servo_moving_unknown_trajectory.py
```
- Use SAVE_GIF = False for development.

Enable GIF export only when finalizing:
- SAVE_GIF = True

Result is saved to:
- assets/sim_visual_servo_moving_circular_trajectory.gif
- assets/sim_visual_servo_unknown_trajectory.gif

---

## 📸 Preview (GIFs)

### 🔵 Visual Servoing — Tracking Circular Target Motion
![Visual Servoing Circular Motion](assets/sim_visual_servo_moving_circular_target.gif)

### 🔴 Visual Servoing — Tracking Unknown / Random Target Motion
![Visual Servoing Unknown Motion](assets/sim_visual_servo_unknown_target.gif)


---

🔥 **Why this project matters in real robotics**

Visual servoing is a core skill of **robotic manipulation** — the ability for a robot to move **based on what the camera sees**, not based on a pre-planned static target.

| System | Real Use |
|--------|----------|
| Industrial arms | Tracking moving conveyor objects |
| Humanoid robots | Handing over objects to humans |
| Home robots | Grasping toys, bottles, tools while they move |
| Research robots | Real-time perception-based control |

🧠 **IK tells:** where the robot must go  
🚀 **Visual servoing tells:** how to *continuously correct and follow* until the object is reached

This project builds intuition for:

- Real-time **camera feedback → control**
- **Closed-loop** joint correction
- **Motion planning under uncertainty**
- **Dynamic target tracking** — no static goal
- **Eye-in-hand perception for manipulation**

---

🏁 **Takeaway**

After completing this project, I now understand:

✔ IK alone is **not enough** for object tracking  
✔ Robots need **continuous perception → correction** to reach moving objects  
✔ Visual servoing merges **camera tracking + IK + smooth joint motion**  
✔ This is the foundation of **autonomous robotic manipulation** and **human-interactive robots**

---
