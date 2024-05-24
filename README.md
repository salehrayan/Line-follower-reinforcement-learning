Simple DQN result simulation (without n-step, Experience replay, etc.) in Webots virtual environment using the Q network that produced the Best Reward in training:

![gif.gif](https://github.com/salehrayan/Line-follower-reinforcement-learning/blob/main/Figures_gifs/simple_DQN_result.gif)

The observations for this E-puck robot were 3 continuous ground sensor readings mounted on this robot that gave higher readings when the ground color was white.
The action space was discrete, i.e. either turn clockwise/counter-clockwise or don't turn and move forward.
Reward history plot:

![figure1.jpg](https://github.com/salehrayan/Line-follower-reinforcement-learning/blob/main/Figures_gifs/best_weights_using_gs_fc_128_100ms_2.jpg)


I created another line following task using the PyBullet package to try making everything python based. The robot is a custom 4-wheeled robot. The observation space is discrete 3 readings indicating if the left, center and right of the front of the robot is black or white. The robot has a constant speed. The action space is a continuous value from -1 to 1 radians controlling the steering. TRPO algorithm is used:

![gif.gif](https://github.com/salehrayan/Line-follower-reinforcement-learning/blob/main/PyBullet_test/Code/TRPO_LineFollowerV0_results_final/TRPO_LineFollowerV0-vid_5.gif)


![demo.mp4](https://github.com/salehrayan/Line-follower-reinforcement-learning/blob/main/PyBullet_test/Code/TRPO_LineFollowerV0_results_final/pybullet_demo.mp4)
