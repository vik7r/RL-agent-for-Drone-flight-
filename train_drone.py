import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# --- CONFIGURATION ---
TOTAL_TIMESTEPS = 100000   # Increase for better performance
MODEL_NAME = "ppo_drone_hover"
RENDER_TEST = True         # Set to True to watch the drone after training

def train():
    """
    Trains the PPO agent in the HoverAviary environment.
    """
    print(f"--- Starting Training ({TOTAL_TIMESTEPS} steps) ---")
    
    # 1. Create the Environment
    # We use HoverAviary: The goal is to stabilize at a target height.
    # obs=KIN: Returns kinematic data (position, velocity, rotation).
    # act=RPM: We control the 4 motor speeds directly.
    env = HoverAviary(obs=ObservationType.KIN, act=ActionType.RPM)
    
    # 2. Vectorize the environment
    # SB3 uses vector environments for efficiency.
    vec_env = make_vec_env(lambda: env, n_envs=1)

    # 3. Initialize the PPO Agent
    model = PPO(
        "MlpPolicy",    # Multi-Layer Perceptron (standard Neural Network)
        vec_env,
        verbose=1,
        learning_rate=0.0003,
        batch_size=64,
        gamma=0.99
    )

    # 4. Train
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    print("--- Training Complete ---")
    
    # 5. Save the Model
    model.save(MODEL_NAME)
    print(f"Model saved as {MODEL_NAME}.zip")
    return model

def test(model):
    """
    Loads the trained model and visualizes the drone's flight.
    """
    print("--- Testing Trained Agent ---")
    
    # Create the environment with GUI enabled for visualization
    env = HoverAviary(obs=ObservationType.KIN, act=ActionType.RPM, gui=RENDER_TEST)
    
    obs, info = env.reset()
    total_reward = 0
    
    # Run a simulation loop
    for _ in range(1000): # Run for 1000 steps
        # Predict the best action using the trained brain
        action, _states = model.predict(obs, deterministic=True)
        
        # Apply action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            obs, info = env.reset()
            
    print(f"Test Flight Complete. Total Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    # Check if a model already exists to avoid retraining every time
    if os.path.exists(f"{MODEL_NAME}.zip"):
        print(f"Found existing model: {MODEL_NAME}.zip. Loading...")
        model = PPO.load(MODEL_NAME)
    else:
        model = train()
    
    # Run the visualization
    test(model)