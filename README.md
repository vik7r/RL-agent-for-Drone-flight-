# AI Drone Navigation Agent using PPO

This repository contains a Reinforcement Learning (RL) agent capable of stabilizing and hovering a quadcopter drone. 

The project utilizes **Proximal Policy Optimization (PPO)**, a state-of-the-art policy gradient method, to teach the drone how to control its four motor speeds (RPM) based on kinematic sensor data.

## 🛠️ Tech Stack

* **Algorithm:** PPO (Proximal Policy Optimization)
* **Library:** Stable Baselines3 (SB3)
* **Simulator:** Gym-Pybullet-Drones (Physics-based engine)
* **Language:** Python 3.8+

## 🚀 Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/drone-rl-agent.git](https://github.com/yourusername/drone-rl-agent.git)
    cd drone-rl-agent
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

Run the main script to train the agent. Once training is complete, a simulation window will open showing the drone's flight.

```bash
python train_drone.py