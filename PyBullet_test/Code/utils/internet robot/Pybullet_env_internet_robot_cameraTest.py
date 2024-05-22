import numpy as np
import pybullet as p
from time import sleep
import pybullet_data
import cv2


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
plane = p.loadURDF('cube.urdf', [0, 0, -0.5 * globalScaling], globalScaling=globalScaling,
                   useFixedBase=1)
p.changeDynamics(plane, -1, lateralFriction=1)
texture_id = p.loadTexture(r'E:\github\Line_follower_test\PyBullet_test\Code\utils\internet robot\track_texture.png')
p.changeVisualShape(
    plane,
    -1,
    textureUniqueId=texture_id, rgbaColor=[1, 1, 1, 1])

# track_waypoints = np.load(r'E:\github\Line_follower_test\PyBullet_test\Code\utils\track_waypoints.npy')
# track_waypoints_z = np.ones((track_waypoints.shape[0], 1)) * 0.05
# track_waypoints = np.hstack((track_waypoints, track_waypoints_z))
# track_waypoints = track_waypoints * np.array([1, -1, 1]) * 0.98
# track_waypoints = track_waypoints.tolist()
# track_waypoints_color = np.ones_like(track_waypoints) * np.array([1, 0, 0])
# p.addUserDebugPoints(track_waypoints, track_waypoints_color)

wheel_indices = [1, 3, 4, 5]
hinge_indices = [0, 2]


def get_camera_image(car_id):
    car_pos, car_orn = p.getBasePositionAndOrientation(car_id)
    car_orn = p.getMatrixFromQuaternion(car_orn)
    car_forward = [car_orn[0], car_orn[3], car_orn[6]]
    car_up = [car_orn[2], car_orn[5], car_orn[8]]

    camera_position = [car_pos[0] + car_forward[0] * 0.08, car_pos[1] + car_forward[1] * 0.08, car_pos[2]+0.17]
    car_cam_target = [car_pos[0] + car_forward[0] * 1.5, car_pos[1] + car_forward[1] * 1.5, car_pos[2]]

    view_matrix = p.computeViewMatrix(camera_position, car_cam_target, car_up)
    proj_matrix = p.computeProjectionMatrixFOV(60, 1, 0.1, 100)
    _, _, px, _, _ = p.getCameraImage(64, 64, view_matrix, proj_matrix)
    rgb_array = np.array(px/255, dtype=np.float32).reshape((64, 64, 4))
    rgb_array = rgb_array[:, :, :3]
    return rgb_array



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

    # Get camera image from the car's perspective
    get_camera_image(car)
    # cv2.imshow('Car Camera', camera_image)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    sleep(1 / 240)

# cv2.destroyAllWindows()
p.disconnect()
