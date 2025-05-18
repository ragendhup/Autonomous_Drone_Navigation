from stable_baselines3 import DQN
from drone_env import DroneEnv

# Load trained model
env = DroneEnv()
model = DQN.load("drone_dqn")

obs = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {done}")

env.close()
