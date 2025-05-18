from stable_baselines3 import DQN  
from drone_env import DroneEnv  

# Create the environment
env = DroneEnv()  

# Initialize the RL model (DQN)
model = DQN("MlpPolicy", env, verbose=1)  

# Train the model
model.learn(total_timesteps=1000)  

# Save the trained model
model.save("drone_dqn")  
print("Training Completed!")  
