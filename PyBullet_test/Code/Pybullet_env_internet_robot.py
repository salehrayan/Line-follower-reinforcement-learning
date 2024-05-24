import numpy as np
import pybullet as p
from time import sleep
import pybullet_data


def rotatel(myList):
    back = myList.pop(0)
    myList.append(back)
    return myList

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)

angle = p.addUserDebugParameter('Steering', -1.57, 1.57, 0)
throttle = p.addUserDebugParameter('Throttle', 0, 20, 0)

globalScaling = 20

car = p.loadURDF('simplecar - Copy.urdf', [0, 0, 0.1])
plane = p.loadURDF('cube.urdf', [0, 0, -0.5 *globalScaling], globalScaling=globalScaling,
                   useFixedBase=1)
p.changeDynamics(plane, -1, lateralFriction=1)
texture_id = p.loadTexture(r'E:\github\Line_follower_test\PyBullet_test\Code\utils\track_texture.png')
p.changeVisualShape(
    plane,
    -1,
    textureUniqueId=texture_id, rgbaColor=[1, 1, 1, 1])


track_waypoints = np.load(r'track_waypoints.npy')
track_waypoints_z = np.ones((track_waypoints.shape[0], 1)) * 0.05
track_waypoints = np.hstack((track_waypoints, track_waypoints_z))
track_waypoints = track_waypoints * np.array([1, -1, 1]) * 0.98
track_waypoints = track_waypoints.tolist()
track_waypoints_color = np.ones_like(track_waypoints) * np.array([1, 0, 0])
p.addUserDebugPoints(track_waypoints, track_waypoints_color)

# for start, end in zip(track_waypoints[0::4], rotatel(track_waypoints)[0::4]):
#     p.addUserDebugLine(start, end, lineColorRGB=[1, 0, 0], lineWidth=10)

wheel_indices = [1, 3, 4, 5]
hinge_indices = [0, 2]

while True:
    user_angle = p.readUserDebugParameter(angle)
    user_throttle = p.readUserDebugParameter(throttle)
    for joint_index in wheel_indices:
        p.setJointMotorControl2(car, joint_index,
                                p.VELOCITY_CONTROL,
                                targetVelocity=user_throttle)
    for joint_index in hinge_indices:
        p.setJointMotorControl2(car, joint_index,
                                p.POSITION_CONTROL,
                                targetPosition=user_angle)
    p.stepSimulation()
    sleep(1/240)

