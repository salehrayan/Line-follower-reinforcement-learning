import pybullet as p
import numpy as np
import time
import pybullet_data
import cv2


physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
# p.setPhysicsEngineParameter(enableConeFriction=0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

robotCarId = p.loadURDF(r"E:\github\Line_follower_test\PyBullet_test\Code\utils\simple_robot_car_1.urdf", [0, 0, 0.5])

# p.changeDynamics(planeId, -1, lateralFriction=4)
# p.changeDynamics(robotCarId, 3, restitution=1000)
joint_indices = {
    "base_to_left_back_wheel": 0,
    "base_to_right_back_wheel": 1,
    "base_to_forward_wheel_hinge": 2,
    "forward_wheel_hinge_to_wheel": 3,
}

# num_joints = p.getNumJoints(robotCarId)
#
# for joint_index in range(num_joints):
#     joint_info = p.getJointInfo(robotCarId, joint_index)
#     print("Joint Index:", joint_info[0])
#     print("Joint Name:", joint_info[1].decode('utf-8'))
#     print("Link Name:", joint_info[12].decode('utf-8'))
#     print("Link Mass:", joint_info[13])
#     print("Link Collision Shape Index:", joint_info[14])
#     print("Link Visual Shape Index:", joint_info[15])
#     print("Link Parent Index:", joint_info[16])
#     print("Link Joint Type:", joint_info[2])
#     print("Link Position in Parent Frame:", joint_info[14:17])
#     print("-" * 30)

cv2.namedWindow('Joint Control')

def nothing(x):
    pass

cv2.createTrackbar('Left Back Wheel', 'Joint Control', 0, 100, nothing)
cv2.createTrackbar('Right Back Wheel', 'Joint Control', 0, 100, nothing)
cv2.createTrackbar('Front Wheel Steering', 'Joint Control', 0, 100, nothing)
cv2.createTrackbar('Front Wheel Rotation', 'Joint Control', 0, 100, nothing)

def map_to_velocity(val):
    return val - 50


while True:
    p.stepSimulation()
    time.sleep(1. / 240.)

    left_back_wheel_vel = (cv2.getTrackbarPos('Left Back Wheel', 'Joint Control'))
    right_back_wheel_vel = left_back_wheel_vel
    front_steering_pos = map_to_velocity(cv2.getTrackbarPos('Front Wheel Steering', 'Joint Control'))
    front_rotation_vel = (cv2.getTrackbarPos('Front Wheel Rotation', 'Joint Control'))

    p.setJointMotorControl2(robotCarId, joint_indices["base_to_left_back_wheel"], p.VELOCITY_CONTROL, force=0)
    p.setJointMotorControl2(robotCarId, joint_indices["base_to_right_back_wheel"], p.VELOCITY_CONTROL, force=0)
    p.setJointMotorControl2(robotCarId, joint_indices["base_to_forward_wheel_hinge"], p.POSITION_CONTROL, targetPosition=front_steering_pos)
    p.setJointMotorControl2(robotCarId, joint_indices["forward_wheel_hinge_to_wheel"], p.VELOCITY_CONTROL, targetVelocity=front_rotation_vel)

    cv2.imshow('Joint Control', np.zeros((1, 500), np.uint8))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    keys = p.getKeyboardEvents()
    if p.B3G_RETURN in keys and keys[p.B3G_RETURN] & p.KEY_WAS_TRIGGERED:
        break

p.disconnect()
cv2.destroyAllWindows()




