# Real-DRL: Experiment-Go2 (Wild Environment)

![PyTorch](https://img.shields.io/badge/PyTorch-3.2.6-red?logo=pytorch)
![Tensorflow](https://img.shields.io/badge/Tensorflow-2.11.0-orange?logo=tensorflow)
![IsaacGym](https://img.shields.io/badge/IsaacGym-Preview4-darkgrey?logo=isaacgym)
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Linux](https://img.shields.io/badge/Linux-22.04-yellow?logo=linux)

---

This repo implements the **Real-DRL** on the Unitree-Go2 robot for the wild environments in Nivida IsaacGym. In this framework, a closed-loop system is established for the quadruped robot by incorporating the *Sensing*, *Navigation/Planning* and *Locomotion Control* module.
<p align="center">
 <img src="./docs/navigation.png" height="460" alt="scene"/> 
 <br><b>Fig 1. Runtime Learning Framework -- A Close-loop System for the Quadruped Robot</b>
</p>

## 💡 User Guide

### ⚙️ Dependencies

* *Python - 3.8 or above*
* *PyTorch - 1.10.0*
* *Isaac Gym - Preview 4*

### 🔨 Setup

1. Clone this repository:

```bash
git clone git@github.com:Charlescai123/isaac-wild-go2.git
```

2. Create the conda environment with:

```bash
conda env create -f environment.yml
```

3. Activate conda environment and Install `rsl_rl` lib:

```bash
conda activate isaac-wild
cd extern/rsl_rl && pip install -e .
```

4. Download and install IsaacGym:

* Download [IsaacGym](https://developer.nvidia.com/isaac-gym) and extract the downloaded file to the root folder.
* navigate to the `isaacgym/python` folder and install it with commands:
* ```bash
  cd isaacgym/python && pip install -e .
  ```
* Test the given example (ignore the error about GPU not being utilized if any):
* ```bash
  cd examples && python 1080_balls_of_solitude.py
  ```

5. Build and install the interface to Unitree's SDK:

* First, install the required packages `Boost` and `LCM`:

   ```bash
   sudo apt install libboost-all-dev liblcm-dev
   ```

* Then, go to `extern/go2_sdk` and create a build folder:
   ```bash
   cd extern/go2_sdk
   mkdir build && cd build
   ```

  Now, build the libraries and move them to the main directory by running:
   ```bash
   cmake ..
   make
   mv go2_interface* ../../..
   ```

## Runtime Learning in the Wild

[//]: # (<p align="center">)
[//]: # ( <img src="./docs/scene.png" alt="rlm"/> )

### 📍 Navigation

**Map Generation Pipeline:** Exterioception (Depth) ➡ BEV Map ➡ Occupancy Map ➡ Cost Map

| BEV Map                                      | Occupancy Map                                      | Cost Map                                     |
|----------------------------------------------|----------------------------------------------------|----------------------------------------------|
| <img src="./docs/bev_map.png" height="330"/> | <img src="./docs/occupancy_map.png" height="330"/> | <img src="./docs/costmap.png" height="330"/> |

The quadruped robot navigates through the wild environment alongside all the waypoints:

```bash
python -m src.scripts.play --use_gpu=True --show_gui=True --num_envs=1
```

| Navigate in the Wild          | RGB Camera Image                           | Depth Camera Image                          |
|-------------------------------|--------------------------------------------|---------------------------------------------|
| <img src="./docs/nav.gif" alt="rlm"/> | <img src="./docs/nav_rgb.gif"  alt="rlm"/> | <img src="./docs/nav_depth.gif" alt="rlm"/> |

### 🦿 Locomotion

---


The locomotion control module provides real-time response in safety-critical systems, effectively
handling unforeseen incidents arising from unknown environments.


#### 1️⃣ Safety Assurance (Runtime Learning)

A key objective of this framework is to ensure the robot's safety during runtime learning, achieved through a
hybrid control system with a switching mechanism design:

🔹 when the robot base turns **Blue** ➡️ robot is controlled by **DRL-Student**.

🔺 when the robot base turns **Red** ➡️ robot is controller by **PHY-Teacher**.

| With Runtime-Learning Framework                         | Without the Framework                                 |
   |---------------------------------------------------------|-------------------------------------------------------|
| <img src="./docs/with-rlm.gif" height="245" alt="rlm"/> | <img src="./docs/wo-rlm.gif" height="245" alt="rlm"/> |

#### ️2️⃣ **Compare with Other Model-based Controller**

PHY-Teacher is a real-time, physics-based safety controller utilizing a dynamic model (**Real-Time Patch**), holding
superior performance compared to safety controllers that rely on time-invariant (e.g., **Fixed**) models. The comparison
on wild, uneven terrain is demonstrated:

| Real-Time Patch (under random push)                         | Fixed Robot Model                                          |
|-------------------------------------------------------------|------------------------------------------------------------|
| <img src="./docs/rlm_go2_push.gif" height="245" alt="rlm"/> | <img src="./docs/fixed_model.gif" height="245" alt="rlm"/> |


## 🏷️ Misc

- In order to plot the latest saved trajectory, run command `python -m src.utils.plot_trajectory`
