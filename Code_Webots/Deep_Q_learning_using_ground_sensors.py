# Add the controller Webots' Python library path
import copy
import sys
webots_controller_path = r'D:\Webots\lib\controller\python'
sys.path.append(webots_controller_path)


from controller import Robot
from controller import Supervisor

import os
import time
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


#Robot Instance
robot = Robot()


#set seed
seed = 42
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda")

class Environment(Supervisor):

    def __init__(self):
        super().__init__()

        self.max_motor_speed = 6
        self.min_motor_speed = 0.5
        # self.basic_time_step = robot.getBasicTimeStep()
        self.sampling_period = int(robot.getBasicTimeStep())

        # Wheel control and status
        self.left_motor = robot.getDevice('left wheel motor')
        self.right_motor = robot.getDevice('right wheel motor')

        # Set the motors to rotate for ever
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        # Zero out starting velocity
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Enable touch sensor
        self.touch = robot.getDevice("touch sensor")
        self.touch.enable(self.sampling_period)

        # Ground sensor list
        self.gs = []
        for i in range(3):
            self.gs.append(robot.getDevice(f"gs{i}"))
            self.gs[-1].enable(self.sampling_period)

        # Enable camera
        # self.camera = robot.devices['camera']
        # self.camera.enable(self.sampling_period)

        # Reset
        self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        robot.step(200)

        self.rotation_punishment_factor = 100


    def get_gs_values(self):
        """

        Returns:
        - numpy.ndarray of 3 ground sensors values at the current time step.
        """

        return np.array([self.gs[0].getValue(), self.gs[1].getValue(), self.gs[2].getValue()], dtype=np.float32)

    def get_agent_observation(self):
        """
        Gets the state/observation that is going to be input into the agent. Here it is the camera image.

        Returns:
            - numpy.ndarray of the camera image frame.
        """

        return np.array([self.gs[0].getValue(), self.gs[1].getValue(), self.gs[2].getValue()], dtype=np.float32)


    def reset(self):
        """
        Resets the environment to its initial state and returns the initial observation.

        Returns:
        - numpy.ndarray: camera_image.
        """

        self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        return self.get_agent_observation()


    def step(self, action, max_steps):
        """
        Takes a step in the environment based on the given action.

        Returns:
        - state       = float numpy.ndarray (width,height,3) of camera image
        - step_reward = float
        - done        = bool
        """
        self.apply_action(action)

        step_reward, done = self.get_reward()

        state = self.get_agent_observation()  # New state

        # Time-based termination condition
        if (int(self.getTime()) + 1) % max_steps == 0:
            done = True

        return state, step_reward, done


    def get_reward(self):
        """
        Calculates and returns the reward based on the current state.

        Returns:
        - The reward and done flag.
        """

        gs_values = self.get_gs_values()
        left_motor_speed = self.left_motor.target_velocity
        right_motor_speed = self.right_motor.target_velocity
        collision = self.touch.getValue()

        # reward = ~(gs_values[0]> 450 and gs_values[1]> 450 and gs_values[2]> 450) * \
        #          ((gs_values[1]> gs_values[0] and gs_values[1] > gs_values[2]) * 100 +\
        #           (gs_values[0]> gs_values[1] and gs_values[0] > gs_values[2]) * 30 +\
        #           (gs_values[2]> gs_values[1] and gs_values[2] > gs_values[0]) * 30) + \
        #          (gs_values[0] > 450 and gs_values[1] > 450 and gs_values[2] > 450) * -2000 + \
        #          (left_motor_speed>0.5)*left_motor_speed**2 + (right_motor_speed>0.5)*right_motor_speed**2 +\
        #     collision * -10000 + (left_motor_speed < 0.5 or right_motor_speed < 0.5) * -150
        # reward = ~(gs_values[0] > 450 and gs_values[1] > 450 and gs_values[2] > 450) * \
        #          ((gs_values[1] > gs_values[0] and gs_values[1] > gs_values[2]) * 70 + \
        #           (gs_values[0] > gs_values[1] and gs_values[0] > gs_values[2]) * 15 + \
        #           (gs_values[2] > gs_values[1] and gs_values[2] > gs_values[0]) * 15) + \
        #          (left_motor_speed > 0.5) * left_motor_speed ** 2 + (right_motor_speed > 0.5) * right_motor_speed ** 2 \
        #          - abs(left_motor_speed - right_motor_speed) * self.rotation_punishment_factor + \
        #          collision * -100 + (left_motor_speed < 0.5 or right_motor_speed < 0.5) * -150

        reward = ~(gs_values[0]> 450 and gs_values[1]> 450 and gs_values[2]> 450) * \
                 ((gs_values[1]> gs_values[0] and gs_values[1] > gs_values[2]) * 1 +\
                  (gs_values[0]> gs_values[1] and gs_values[0] > gs_values[2]) * 0.25 +\
                  (gs_values[2]> gs_values[1] and gs_values[2] > gs_values[0]) * 0.25) + \
                 (gs_values[0] > 450 and gs_values[1] > 450 and gs_values[2] > 450) * -1 + \
                 (left_motor_speed + right_motor_speed)/12 +\
            collision * -1

        done = bool(int(collision))
        # if (gs_values[0] > 450 and gs_values[1] > 450 and gs_values[2] > 450): reward += -500

        return reward, done

    def apply_action(self, action):
        """
        Applies the specified action to the robot's motors.

        Returns:
        - None
        """
        left_motor_speed = self.left_motor.target_velocity
        right_motor_speed = self.right_motor.target_velocity
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        match action:
            case 0:
                self.right_motor.setVelocity(0)
                self.left_motor.setVelocity(self.max_motor_speed)
            case 1:
                self.left_motor.setVelocity(0)
                self.right_motor.setVelocity(self.max_motor_speed)
            case 2:
                self.left_motor.setVelocity(5)
                self.right_motor.setVelocity(5)

        self.left_motor.setVelocity((self.left_motor.target_velocity > 0)* self.left_motor.target_velocity)
        self.right_motor.setVelocity((self.right_motor.target_velocity > 0) * self.right_motor.target_velocity)

        robot.step(100)
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

class NETWORK(torch.nn.Module):
    """Neural network model representing the policy network."""

    def __init__(self, kernel_size):
        super(NETWORK, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size)
        self.conv2 = nn.Conv2d(32, 64, kernel_size)
        self.fc1 = nn.Linear(64,128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        """Performs the forward pass through the network and computes action probabilities."""

        x = torch.relu(self.conv1(x))
        x = nn.functional.avg_pool2d(torch.relu(self.conv2(x)), 124).squeeze(2).squeeze(2)
        x = self.fc3(nn.functional.relu(self.fc2(nn.functional.relu(self.fc1(x)))))

        return x


# q_network = NETWORK(3).to(device)
q_network = nn.Sequential(nn.Linear(3, 128),
                          nn.ReLU(),
                          nn.Linear(128,128),
                          nn.ReLU(),
                          nn.Linear(128,3)).to(device)
target_q_network = copy.deepcopy(q_network).eval().to(device)

def e_greedy_policy(state, epsilon=0.):
    if torch.rand(1) < epsilon:
        return torch.randint(3, (1,1))
    else:
        action_values = q_network(state).detach()
        return torch.argmax(action_values, dim=-1, keepdim=True)

class Agent_DQN():
    """Agent implementing the REINFORCE algorithm."""

    def __init__(self, save_path, load_path, num_episodes, max_steps,
                 learning_rate, gamma, clip_grad_norm, baseline):
        self.save_path = save_path
        self.load_path = load_path

        # os.makedirs(self.save_path, exist_ok=True)

        # Hyper-parameters Attributes
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.learing_rate = learning_rate
        self.gamma = gamma
        self.clip_grad_norm = clip_grad_norm
        self.baseline = baseline

        self.optimizer = optim.AdamW(q_network.parameters(), self.learing_rate)

        # instance of env
        self.env = Environment()


    def save(self):
        """Save the trained model parameters after final episode and after receiving the best reward."""
        torch.save(q_network.state_dict(), self.save_path)

    def save_reward_plot(self, path, rewards):
        """Save reward plot each episode"""
        rewards = np.array(rewards)
        sma = np.convolve(rewards, np.ones(25) / 25, mode='valid')

        plt.figure()
        plt.title("Episode Rewards")
        plt.plot(rewards, label='Raw Reward', color='#142475', alpha=0.45)
        plt.plot(sma, label='SMA 25', color='#f0c52b')
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)

        plt.savefig(self.save_path[:-3] + '.png', format='png', dpi=1000, bbox_inches='tight')
    def load(self):
        """Load pre-trained model parameters."""
        q_network.load_state_dict(torch.load(self.load_path))


    def compute_loss(self, state, action, reward, next_state, done):
        """
            Compute the Q-Learning loss.
        """

        target_qsa = reward + (~done)* self.gamma* torch.max(target_q_network(torch.from_numpy(next_state).to(device).unsqueeze(0)))
        before_qsa = q_network(torch.from_numpy(state).to(device).unsqueeze(0)).squeeze(0)[action[0][0]]
        return nn.functional.mse_loss(target_qsa, before_qsa)

    def train(self):
        """
        Train the agent using the DQN algorithm.

        """
        q_network.train()
        start_time = time.time()
        reward_history = []
        best_score = -np.inf

        for episode in range(1, self.num_episodes + 1):
            state = self.env.reset()

            rewards = []
            ep_reward = 0


            time_step = 0
            while True:
                print(f'\rtime step {time_step}/{self.max_steps}', end='')
                action = e_greedy_policy(torch.from_numpy((state-np.mean(state))/(np.std(state)+0.01)).to(device).unsqueeze(0), epsilon=0.15)
                next_state, reward, done = self.env.step(action.item(), self.max_steps)

                rewards.append(reward)
                ep_reward += reward
                time_step += 1

                done = done or (time_step == self.max_steps)

                loss = self.compute_loss(((state-np.mean(state))/(np.std(state)+0.01)), action,
                                         reward, ((next_state-np.mean(next_state))/(np.std(next_state)+0.01)), done)
                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(q_network.parameters(), self.clip_grad_norm)
                self.optimizer.step()

                # print(q_network(torch.from_numpy((state-np.mean(state))/(np.std(state)+0.01)).to(device).unsqueeze(0)).detach())
                state = next_state


                if done:
                    break

            reward_history.append(ep_reward)
            if ep_reward > best_score:
                self.save()
                best_score = ep_reward
            print(f"Episode {episode}: Score = {ep_reward:.3f}")

            if episode%50 == 0:
                self.save_reward_plot(self.save_path, reward_history)

            if episode % 10 == 0:
                target_q_network.load_state_dict(q_network.state_dict())

        # Save final weights and plot reward history
        # self.save(path='/final_weights.pt')
        # self.plot_rewards(reward_history)

        # Print total training time
        elapsed_time = time.time() - start_time
        elapsed_timedelta = timedelta(seconds=elapsed_time)
        formatted_time = str(elapsed_timedelta).split('.')[0]
        print(f'Total Spent Time: {formatted_time}')

    def test(self):
        """
        Test the trained agent.
        This method evaluates the performance of the trained agent.
        """

        start_time = time.time()
        rewards = []
        self.load()
        q_network.eval()

        for episode in range(1, self.num_episodes+1):
            state = self.env.reset()
            done = False

            ep_reward = 0
            while not done:
                action = torch.argmax(q_network(torch.from_numpy((state-np.mean(state))/(np.std(state)+0.01)).to(device).unsqueeze(0)), dim=-1)
                # if torch.rand(1) < 0.05:
                #     action = torch.randint(3, (1, 1))
                #     print('now')
                # else:
                #     action = torch.argmax(q_network(
                #         torch.from_numpy((state - np.mean(state)) / (np.std(state) + 0.01)).to(device).unsqueeze(0)),
                #                           dim=-1)

                next_state, reward, done = self.env.step(action.item(), self.max_steps)
                ep_reward += reward
                state = next_state
            rewards.append(ep_reward)
            print(f"Episode {episode}: Score = {ep_reward:.3f}")
        print(f"Mean Score = {np.mean(rewards):.3f}")

        elapsed_time = time.time() - start_time
        elapsed_timedelta = timedelta(seconds=elapsed_time)
        formatted_time = str(elapsed_timedelta).split('.')[0]
        print(f'Total Spent Time: {formatted_time}')

    def plot_rewards(self, rewards):
        # Calculate the Simple Moving Average (SMA) with a window size of 25
        sma = np.convolve(rewards, np.ones(25) / 25, mode='valid')

        plt.figure()
        plt.title("Episode Rewards")
        plt.plot(rewards, label='Raw Reward', color='#142475', alpha=0.45)
        plt.plot(sma, label='SMA 25', color='#f0c52b')
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)

        plt.savefig(self.save_path[:-3] + '.png', format='png', dpi=1000, bbox_inches='tight')


        plt.show()
        plt.clf()
        plt.close()


save_path = './Deep_Q_learning_results/best_weights_using_gs_fc_128_100ms_2.pt'
load_path = './Deep_Q_learning_results/best_weights_using_gs_fc_128_100ms_2.pt'
train_mode = False
num_episodes = 3000 if train_mode else 10
max_steps = 1000 if train_mode else 500
learning_rate = 1e-4
gamma = 0.99
# hidden_size = 6
clip_grad_norm = 5
baseline = True

agent = Agent_DQN(save_path, load_path, num_episodes, max_steps,
                        learning_rate, gamma, clip_grad_norm, baseline)

if train_mode:
    # Initialize Training
    agent.train()
else:
    # Test
    agent.test()


