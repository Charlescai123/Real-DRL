# Real-DRL: Experiment-A1 (Real-world)

![Tensorflow](https://img.shields.io/badge/Tensorflow-2.5.0-orange?logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Linux](https://img.shields.io/badge/Linux-22.04-yellow?logo=linux)
![Pybullet](https://img.shields.io/badge/Pybullet-3.2.6-brightgreen)

This repository contains the implementation of real-world experiments on the Unitree A1 robot to validate the practical
effectiveness of the Real‑DRL framework.

## Environment Setup

### Setup for Local PC

It is recommended to create a separate virtualenv or conda environment to avoid conflicting with existing system
packages.

Follow the steps below to build the Python environment:

1. First, create the conda environment by running:

   ```bash
   conda create --name real-a1 python==3.11.0 
   ```

2. Second, Install all dependent packages by running:

   ```bash
   python setup.py install
   ```

3. Lastly, build and install the interface to Unitree's SDK. The
   Unitree [repo](https://github.com/unitreerobotics/unitree_legged_sdk) keeps releasing new SDK versions. For
   convenience, we have included the version that we used in `third_party/unitree_legged_sdk`.

   First, make sure the required packages are installed, following
   Unitree's [guide](https://github.com/unitreerobotics/unitree_legged_sdk?tab=readme-ov-file#dependencies). Most
   nostably, please make sure to
   install `Boost` and `LCM`:

   ```bash
   sudo apt install libboost-all-dev liblcm-dev
   ```

   Then, go to `third_party/unitree_legged_sdk` and create a build folder:
   ```bash
   cd third_party/unitree_legged_sdk
   mkdir build && cd build
   ```

   Now, build the libraries and move them to the main directory by running:
   ```bash
   cmake ..
   make
   mv robot_interface* ../../..
   ```

### Setup for Real Robot

Follow the steps if you want to run controllers on the real robot:

1. **Setup correct permissions for non-sudo user (optional)**

   Since the Unitree SDK requires memory locking and high process priority, root priority with `sudo` is usually
   required to execute commands. To run the SDK without `sudo`, write the following
   to `/etc/security/limits.d/90-unitree.conf`:

   ```bash
   <username> soft memlock unlimited
   <username> hard memlock unlimited
   <username> soft nice eip
   <username> hard nice eip
   ```

   Log out and log back in for the above changes to take effect.

2. **Connect to the real robot**

   Configure the wireless on the real robot with the [manual](docs/A1_Wireless_Configuration.pdf), and make sure
   you can ping into the robot's low-level controller (IP:`192.168.123.10`) on your local PC.

## Real-DRL

#### Run script for real-world experiment

```Shell
python job/run_edge_control_real.py
```

#### Run script for simulation experiment

```Shell
python job/run_edge_control.py
```

#### Run script for cloud training for simulation or real-world experiment

```Shell
python job/run_cloud_train.py
```

#### Run script for patch update

```Shell
python job/run_patch_update.py
```

## Cautions

Please exercise caution when running this code on a real robot, as the A1 motor is prone to damage and the framework’s
real-time performance is highly dependent on the computing platform. We previously encountered hardware issues during
testing. To minimize risk, we strongly recommend thoroughly validating both the code and the hardware setup (e.g., the
host machine’s CPU and GPU) before deployment.

## Trouble-shootings

In case your onboard motor damaged due to unknown problems, refer to
the [instruction manual](docs/A1_Motor_Replacement.pdf) for its replacement