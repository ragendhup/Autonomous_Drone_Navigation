import gym
import airsim
import numpy as np
from gym import spaces

class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
        
        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()

        # Set goal position (change as needed)
        position = self.client.getMultirotorState().kinematics_estimated.position
        self.goal_x, self.goal_y, self.goal_z = [8, -6, -0.07]
        print("The goal position: ",self.goal_x, self.goal_y, self.goal_z)
        # Define action space: Forward, Left, Right, Up, Down, Hover
        self.action_space = spaces.Discrete(5)

        # Observation space: [x, y, z, lidar_front, lidar_left, lidar_right]
        self.observation_space = spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)

    def step(self, action):
        # Execute the action
        if action == 0:  # Move Forward
            self.client.moveByVelocityAsync(2, 0, 0, 1).join()
        elif action == 1:  # Move Left
            self.client.moveByVelocityAsync(0, -2, 0, 1).join()
        elif action == 2:  # Move Right
            self.client.moveByVelocityAsync(0, 2, 0, 1).join()
        elif action == 3:  # Move Up
            self.client.moveByVelocityAsync(0, 0, -2, 1).join()
        elif action == 4:  # Move Down
            self.client.moveByVelocityAsync(0, 0, 2, 1).join()

        # Get drone position
        position = self.client.getMultirotorState().kinematics_estimated.position
        x, y, z = position.x_val, position.y_val, position.z_val
        print("The current step position: ",x,y,z)
        # Get real-time LiDAR data
        lidar_data = self.client.getLidarData()
        
        if lidar_data is not None and len(lidar_data.point_cloud) > 3:
            points = np.array(lidar_data.point_cloud, dtype=np.float32)
            points = points.reshape(-1, 3)  # Reshape into (N, 3)
            
            # Extract distances (simplified: front, left, right)
            lidar_front = np.min(points[:, 0])  # Closest object in front
            lidar_left = np.min(points[:, 1])   # Closest object on the left
            lidar_right = np.max(points[:, 1])  # Closest object on the right
        else:
            lidar_front, lidar_left, lidar_right = 100, 100, 100  # Default if no data

        # Compute distance to goal
        distance_to_goal = np.sqrt((x - self.goal_x) ** 2 + (y - self.goal_y) ** 2 + (z - self.goal_z) ** 2)

        # Reward system
        reward = -distance_to_goal  # Encourage getting closer to the goal

        # Collision detection
        collision = self.client.simGetCollisionInfo()
        if collision.has_collided:
            print("💥 Collision detected! Ending episode.")
            reward -= 50
            done = True
        else:
            done = False

        # Avoid obstacles using LiDAR
        if lidar_front < 3:  # If obstacle is too close in front
            reward -= 20  # Big penalty for being close to an obstacle
        if lidar_left < 3 or lidar_right < 3:  # If obstacles on the sides
            reward -= 10  

        # **Success condition fix** (Ensure the drone stops)
        print("distance to target is :  ",distance_to_goal)
        if distance_to_goal < 3:
            reward += 100  # Big reward for reaching the goal
            done = True  # **Force episode to stop**
            print(f"🎯 Target reached at ({x}, {y}, {z})! Stopping episode.")

        return np.array([x, y, z, lidar_front, lidar_left, lidar_right], dtype=np.float32), reward, done, {}


    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        return np.array([0, 0, 0, 100, 100, 100], dtype=np.float32)

    def render(self, mode='human'):
        pass

    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
