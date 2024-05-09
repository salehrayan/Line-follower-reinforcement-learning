import sys

webots_path = r'D:\Webots\lib\controller\python'
sys.path.append(webots_path)

from controller import Robot


# TIME_STEP = 64

robot = Robot()

leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')

leftMotor.setPosition(10.0)
rightMotor.setPosition(10.0)

while True:
    robot.step()
