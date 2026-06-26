# 🕷️ SpiderBot

> **A 12-DOF Quadrupedal Robot**
> *Daniele Palomba & Giulio Vestri* — Final Project for Advanced Robotics, University of Naples Federico II.

---

## 🚀 Project Overview

**SpiderBot** is a quadrupedal robot with **12 degrees of freedom (DOF)**. The physical architecture is an evolution of an original hexapod design created by *emrekalem*. We structurally modified and adapted the original files into a quadrupedal configuration, maintaining a regular hexagonal geometry to ensure a robust and scalable control architecture. 

The project covers the full mechatronic stack: from geometric design to simulation and training via end-to-end **Deep Reinforcement Learning (DRL)**, through to physical implementation on real hardware.

### ✨ Key Features

- **Optimized Hexagonal Geometry:** Regular hexagonal-symmetry chassis providing an exceptionally wide and stable static support base.
- **Dual Control Paradigm:**
  - *Open-Loop Kinematics:* Time-varying analytical gait generator (Creep, Trot, Gallop, Pace).
  - *End-to-End Neural Controller:* Control based on DRL policies learned directly in accelerated physics simulation.
- **High-Performance Simulation:** Pipeline based on **JAX, Brax, and MuJoCo MJX** for massively parallel training on GPU/CPU.
- **Efficient Hardware:** Master-Slave architecture based on an Arduino MCU and PCA9685 PWM driver via I2C for synchronous management of 12 MZ996R servomotors.

---

## 🛠️ Control Architecture

### 1. Open-Loop Kinematic Control

The classical kinematic pipeline, implemented in MATLAB/Simulink, is structured on three hierarchical levels:

1. **High-level Trajectory Planner:** Computes the desired 3D Cartesian coordinates of each foot relative to the robot body frame.
2. **Mid-level Spatial Transformation Layer:** Converts Cartesian footprints into joint angles via a geometric Inverse Kinematics (IK) solver.
3. **Low-level Actuation Layer:** Sends PWM commands to individual servomotors (Coxa, Femur, Tibia) while synchronizing the digital twin.

#### 🐾 Implemented Gaits

| Gait | Type | Foot Scheduling Parameters | Description |
| :--- | :--- | :--- | :--- |
| **Creep Gait** | 4-beat | `[0, 25, 50, 75]` (T = 2.0s) | Maximum stability on uneven terrain. |
| **Trot Gait** | 2-beat | `[0, 50, 50, 0]` (T = 0.6s) | Medium-to-high speed gait with synchronized diagonal pairs. |
| **Gallop Gait** | High speed | Pulse synchronization | Maximum dynamic reactivity. |
| **Pace Gait** | 2-beat | Lateral synchronization | Coordinated movement of limbs on the same side. |

### 2. Deep Reinforcement Learning (End-to-End)

As an alternative to classical control, SpiderBot implements an **End-to-End Neural Controller** paradigm. The neural network maps the physical state directly to joint angle targets.

- **Observation Space (35D):** Includes orientation (quaternions), angular velocity, gait phase, and current joint positions.
- **Action Space (12D):** Angular targets in $[-2.5, 2.5]$ rad for the 12 motors.
- **Teleoperated Learning Reward:** Multi-axis tracking optimization via exponential Gaussian reward functions, with dedicated penalties to eliminate lateral drift ($v_x$).

---

## 💻 Repository Structure

```text
├── Matlab/              # Simulink open-loop models and related graphs
├── MuJoCo-Jax/          # DRL training pipeline, enjoy_env, export_json, policies, and XML model
├── Mujoco-Jax-TEL/      # Telemetry, enjoy_env, export_json scripts, policies, and XML model
├── Spiderbot/           # Arduino firmware for the physical robot and MATLAB script for teleoperation
└── Spiderbot_urdf/      # URDF representation of the robot
```

---

## 🔌 Electronics & Hardware Architecture

The physical system is based on a **Master-Slave** topology to optimize computation cycles:

- **External Host (Master):** Handles high-level command streaming or execution of the neural policy.
- **Local MCU (Slave — Arduino):** Receives serial data, computes gait phases, and performs Inverse Kinematics geometry in real time.
- **PCA9685 PWM Driver:** Dedicated I2C actuator that offloads MCU timers and drives the **12 MZ996R servomotors** with precise 12-bit pulses.
- **Chassis:** Entirely 3D-printed using FDM technology (PLA), matching the kinematic dimensions of the URDF model down to the millimeter.

---

## 🚀 Getting Started

### 1. Kinematic Simulation (MATLAB)

1. Open MATLAB and navigate to the `MatLab/` folder.
2. Run `graphs.m` to initialize the gait variables.
3. Open and launch the Simulink model `open_loop.slx`.

### 2. DRL Training (Python/JAX)

Make sure you have an environment with CUDA support to take advantage of MJX:

```bash
cd MuJoCo-Jax
pip install -r requirements.txt
python train_spider.py
```

To visualize results or deploy a pre-trained policy:

```bash
python export_json.py 
python enjoy_spider.py 
```

---

*Project developed in collaboration with the **PRISMA Lab**.*
