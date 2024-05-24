import numpy as np
import pybullet_data
import pybullet as p
import cv2
from time import sleep
from pybullet_utils import bullet_client
from utils import *
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecNormalize, VecTransposeImage, VecEnv
from stable_baselines3.common.monitor import Monitor


import numpy as np
import pybullet_data
import pybullet as p
import cv2
from time import sleep
from pybullet_utils import bullet_client
from utils import *
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecNormalize, VecTransposeImage, VecEnv
from stable_baselines3.common.monitor import Monitor


class LineFollowerV0(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, max_steps, render_mode='rgb_array'):
        super().__init__()

        self.client = bullet_client.BulletClient(connection_mode=p.DIRECT)
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.client.setGravity(0, 0, -10)

        globalScaling = 20

        self.car = self.client.loadURDF('simplecar - Copy.urdf', [0, 0, 0.1])
        _, self.car_orientation = self.client.getBasePositionAndOrientation(self.car)
        self.plane = self.client.loadURDF('cube.urdf', [0, 0, -0.5 * globalScaling], globalScaling=globalScaling, useFixedBase=1)
        self.client.changeDynamics(self.plane, -1, lateralFriction=1)
        texture_id = self.client.loadTexture(r'E:\github\Line_follower_test\PyBullet_test\Code\utils\track_texture.png')
        self.client.changeVisualShape(self.plane, -1, textureUniqueId=texture_id, rgbaColor=[1, 1, 1, 1])

        self.wheel_indices = [1, 3, 4, 5]
        self.hinge_indices = [0, 2]

        self.render_mode = render_mode
        self.observation_space = spaces.MultiDiscrete([2, 2, 2], dtype=np.int32)
        self.action_space = spaces.Discrete(3)
        self.max_steps = max_steps
        self.n_step = 0

        self.waypoint_monitor = WaypointsMonitor(client=self.client, waypoints_path='track_waypoints.npy',
                                                 car=self.car)

    def reset(self, seed=None, options=None):
        # super().reset(seed=seed)

        # starting_pos = self.np_random.choice(self.waypoint_monitor.waypoints_monitor_for_reset, size=1).squeeze()
        self.client.resetBasePositionAndOrientation(self.car,
                                                    (-3.536647007992482, 0.4236722300659929, 0.09807812190507681),
                                                    (-0.0014877329980157537, -0.0016113340695280331, -0.676497446239607,
                                                     0.7364417122110434))

        self.waypoint_monitor.reset()
        self.n_step = 0

        observation = get_camera_image(self.car, self.client)
        info = {}
        self.angle = 0
        for joint_index in self.wheel_indices:
            self.client.setJointMotorControl2(self.car, joint_index, self.client.VELOCITY_CONTROL, targetVelocity=3)
        return observation, info

    def step(self, action):


        if action == 0:
            self.angle = 1
            for joint_index in self.hinge_indices:
                self.client.setJointMotorControl2(self.car, joint_index, self.client.POSITION_CONTROL,
                                                  targetPosition=self.angle)
        elif action == 1:
            self.angle = -1
            for joint_index in self.hinge_indices:
                self.client.setJointMotorControl2(self.car, joint_index, self.client.POSITION_CONTROL,
                                                  targetPosition=self.angle)
        else:
            self.angle = 0
            for joint_index in self.hinge_indices:
                self.client.setJointMotorControl2(self.car, joint_index, self.client.POSITION_CONTROL,
                                                  targetPosition=self.angle)

        for _ in range(int(0.2 / (1 / 240))):
            self.client.stepSimulation()

        closest_waypoint_distance, passed = self.waypoint_monitor.step_monitor(threshold=0.2)

        observation = get_camera_image(self.car, self.client)
        self.n_step += 1

        terminated_good = (self.waypoint_monitor.num_waypoints_passed == self.waypoint_monitor.num_waypoints_total)

        car_pos, _ = self.client.getBasePositionAndOrientation(self.car)
        terminated_bad = (closest_waypoint_distance > 3)
        truncated = (self.n_step == self.max_steps)

        passed_ratio = self.waypoint_monitor.num_waypoints_passed / self.waypoint_monitor.num_waypoints_total
        reward = ( ((1 - observation[1]) * 5 ) + ((1 - observation[0]) * 1 ) + ((1 - observation[2]) * 1 ) -
                  observation[0] * observation[1] * observation[2] * 1)

        terminated = terminated_bad or terminated_good
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'rgb_array':
            return get_custom_camera_image(self.client)

    def close(self):
        self.client.disconnect()

dir_path = 'E:/github/Line_follower_test/PyBullet_test/Code/PPO_LineFollowerV0_results_2/'

env = LineFollowerV0(max_steps=2000)
# env2 = LineFollowerV0(max_steps=2000)
# env3 = LineFollowerV0(max_steps=2000)
# env4 = LineFollowerV0(max_steps=2000)
# eval_env = LineFollowerV0(max_steps=2000)

# env = Monitor(env)
# eval_env = LineFollowerV0(max_steps=3000)
vec_env = DummyVecEnv([lambda: env])
vec_env = VecVideoRecorder(vec_env, dir_path, record_video_trigger=lambda x: x == 0,
                           video_length=500, name_prefix='PPO_LineFollowerV0')

model = DQN.load(r'C:\Users\ASUS\Downloads/best_model.zip')

obs = vec_env.reset()

for _ in range(500+1):
    action = model.predict(obs)
    next_obs, _, _, _ = vec_env.step(action)
    obs = next_obs
vec_env.close()
