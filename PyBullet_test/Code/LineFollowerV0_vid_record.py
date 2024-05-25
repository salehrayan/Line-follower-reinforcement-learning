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
from sb3_contrib import TRPO
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

        self.client = bullet_client.BulletClient(connection_mode=p.GUI)
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
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
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
            self.client.setJointMotorControl2(self.car, joint_index, self.client.VELOCITY_CONTROL, targetVelocity=4)
        return observation, info

    def step(self, action):

        # user_throttle = (action[0] + 1) * 10
        user_angle = action[0] * 1

        for joint_index in self.hinge_indices:
            self.client.setJointMotorControl2(self.car, joint_index, self.client.POSITION_CONTROL,
                                              targetPosition=user_angle)

        for _ in range(int(0.1 / (1 / 240))):
            self.client.stepSimulation()
        sleep(3/240)

        closest_waypoint_distance, passed = self.waypoint_monitor.step_monitor(threshold=0.2)

        p.addUserDebugPoints(self.waypoint_monitor.waypoints_monitor,
                             np.ones_like(self.waypoint_monitor.waypoints_monitor) * np.array([1, 0, 0]), lifeTime=0.5)
        p.addUserDebugText(f'clossest waypoint: {closest_waypoint_distance}\npassed:{passed}', [1, 1, 1], lifeTime=0.5)

        observation = get_camera_image(self.car, self.client)
        self.n_step += 1

        terminated_good = (self.waypoint_monitor.num_waypoints_passed == self.waypoint_monitor.num_waypoints_total)

        terminated_bad = (closest_waypoint_distance > 1.5)
        truncated = (self.n_step == self.max_steps)

        passed_ratio = self.waypoint_monitor.num_waypoints_passed / self.waypoint_monitor.num_waypoints_total
        reward = (((1 - observation[1]) * 0.5) + ((1 - observation[0]) * 0.1) + ((1 - observation[2]) * 0.1) -
                  observation[0] * observation[1] * observation[2] * 0.1 - closest_waypoint_distance ** 2 -
                  terminated_bad * 50)

        terminated = terminated_bad or terminated_good
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'rgb_array':
            return get_custom_camera_image(self.client)

    def close(self):
        self.client.disconnect()

dir_path = 'E:/github/Line_follower_test/PyBullet_test/Code/PPO_LineFollowerV0_results_2/'

env = LineFollowerV0(max_steps=5000)
# env2 = LineFollowerV0(max_steps=2000)
# env3 = LineFollowerV0(max_steps=2000)
# env4 = LineFollowerV0(max_steps=2000)
# eval_env = LineFollowerV0(max_steps=2000)

# env = Monitor(env)
# eval_env = LineFollowerV0(max_steps=3000)
vec_env = DummyVecEnv([lambda: env])
# vec_env = VecVideoRecorder(vec_env, dir_path, record_video_trigger=lambda x: x == 0,
#                            video_length=500, name_prefix='PPO_LineFollowerV0')

model = TRPO.load(r'TRPO_LineFollowerV0_results_final/best_model.zip')

obs = vec_env.reset()

for _ in range(3000+1):
    action = model.predict(obs)
    next_obs, _, _, _ = vec_env.step(action)
    obs = next_obs
vec_env.close()

