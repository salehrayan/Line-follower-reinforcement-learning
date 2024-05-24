import numpy as np
import pybullet_data
import pybullet as p
import cv2
from time import sleep
from pybullet_utils import bullet_client
from utils import *

# Create a BulletClient instance
client = bullet_client.BulletClient(connection_mode=p.GUI)
client.setAdditionalSearchPath(pybullet_data.getDataPath())
client.setGravity(0, 0, -10)

angle = client.addUserDebugParameter('Steering', -1.57, 1.57, 0)
throttle = client.addUserDebugParameter('Throttle', 0, 20, 0)

globalScaling = 20

car = client.loadURDF('simplecar - Copy.urdf', (-3.536647007992482, 0.4236722300659929, 0.09807812190507681),
                                                    (-0.0014877329980157537, -0.0016113340695280331, -0.676497446239607,
                                                     0.7364417122110434))
plane = client.loadURDF('cube.urdf', [0, 0, -0.5 * globalScaling], globalScaling=globalScaling, useFixedBase=1)
client.changeDynamics(plane, -1, lateralFriction=1)
texture_id = client.loadTexture(r'E:\github\Line_follower_test\PyBullet_test\Code\utils\track_texture.png')
client.changeVisualShape(plane, -1, textureUniqueId=texture_id, rgbaColor=[1, 1, 1, 1])

wheel_indices = [1, 3, 4, 5]
hinge_indices = [0, 2]


waypoints_monitor = WaypointsMonitor(client=client, car=car, waypoints_path='track_waypoints.npy')

while True:
    user_angle = client.readUserDebugParameter(angle)
    user_throttle = client.readUserDebugParameter(throttle)
    for joint_index in wheel_indices:
        client.setJointMotorControl2(car, joint_index, client.VELOCITY_CONTROL, targetVelocity=user_throttle)
    for joint_index in hinge_indices:
        client.setJointMotorControl2(car, joint_index, client.POSITION_CONTROL, targetPosition=user_angle)

    for _ in range(int(0.1 / (1 / 240))):
        client.stepSimulation()

    a = get_camera_image(car, client)
    # get_custom_camera_image(client)

    sleep(6/240)
    clossest, passed = waypoints_monitor.step_monitor(threshold=0.2)

    p.addUserDebugPoints(waypoints_monitor.waypoints_monitor,
                         np.ones_like(waypoints_monitor.waypoints_monitor) * np.array([1, 0, 0]), lifeTime=0.5)
    p.addUserDebugText(f'clossest waypoint: {clossest}\npassed:{passed}', [1,1,1], lifeTime=0.5)

client.disconnect()
