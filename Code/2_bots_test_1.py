import sys
import os

webots_controller_path = r'D:\Webots\lib\controller\python'
sys.path.append(webots_controller_path)
# webots_path = r'D:\Webots'# sys.path.append(webots_path)

from controller import Robot, Camera


# TIME_STEP = 64

robot = Robot()



while True:
    robot.step()

