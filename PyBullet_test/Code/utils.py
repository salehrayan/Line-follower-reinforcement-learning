import numpy as np
import pybullet_data
import pybullet as p
import cv2
from time import sleep
from pybullet_utils import bullet_client


def rotatel(myList):
    back = myList.pop(0)
    myList.append(back)
    return myList

def get_camera_image(car_id, client):
    car_pos, car_orn = client.getBasePositionAndOrientation(car_id)
    car_orn = client.getMatrixFromQuaternion(car_orn)
    car_forward = [car_orn[0], car_orn[3], car_orn[6]]
    car_up = [car_orn[2], car_orn[5], car_orn[8]]

    camera_position = [car_pos[0] + car_forward[0] * 0.2, car_pos[1] + car_forward[1] * 0.2, car_pos[2] + 0.5]
    car_cam_target = [car_pos[0] + car_forward[0] * 1, car_pos[1] + car_forward[1] * 1, car_pos[2]-0.5]

    view_matrix = client.computeViewMatrix(camera_position, car_cam_target, car_up)
    proj_matrix = client.computeProjectionMatrixFOV(60, 1, 0.1, 100)
    _, _, px, _, _ = client.getCameraImage(32, 32, view_matrix, proj_matrix)
    rgb_array = np.array(px, dtype=np.uint8).reshape((32, 32, 4))
    gray_array = cv2.cvtColor(rgb_array[:, :, :3], cv2.COLOR_RGB2GRAY).reshape((32, 32, 1))
    left, center, right = (int(bool(gray_array[-2,8,0])), int(bool(gray_array[-2,16,0]) or bool(gray_array[-2,17,0])),
                           int(bool(gray_array[-2,24,0])))
    return np.array([left, center, right], dtype=np.int32)


def get_custom_camera_image(client):

    armTargetPos = [-0.26, 0.24, -5.77]
    dist = 18.58
    pitch = -80.80
    yaw = 0.40
    roll = 0
    up_axis_index = 2

    view_matrix = client.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=armTargetPos,
        distance=dist,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        upAxisIndex=up_axis_index
    )
    proj_matrix = client.computeProjectionMatrixFOV(60, 1, 0.1, 100)
    _, _, px, _, _ = client.getCameraImage(512, 512, view_matrix, proj_matrix)
    rgb_array = np.array(px, dtype=np.uint8).reshape((512, 512, 4))
    rgb_array = rgb_array[:, :, :3]
    return rgb_array


class WaypointsMonitor():
    def __init__(self, client, waypoints_path, car):
        self.client = client
        self.car = car

        track_waypoints = np.load(waypoints_path)
        track_waypoints_z = np.ones((track_waypoints.shape[0], 1)) * 0.1
        track_waypoints = np.hstack((track_waypoints, track_waypoints_z))
        track_waypoints = track_waypoints * np.array([1, -1, 1]) * 0.98
        # waypoints_monitor_vector = np.zeros((track_waypoints.shape[0], 1))

        # self.waypoints_monitor = np.hstack((track_waypoints, waypoints_monitor_vector))
        self.waypoints_monitor = track_waypoints
        self.waypoints_monitor_for_reset = np.copy(self.waypoints_monitor)
        self.num_waypoints_passed = 0
        self.num_waypoints_total = len(track_waypoints)

    def step_monitor(self, threshold=0.1):
        car_pos_tuple, _ = self.client.getBasePositionAndOrientation(self.car)
        car_pos = np.column_stack(car_pos_tuple)

        distances = np.linalg.norm(self.waypoints_monitor[:, :] - car_pos, axis=1)
        passed_now_indices = (distances < threshold).nonzero()[0]

        self.num_waypoints_passed += len(passed_now_indices)
        self.waypoints_monitor = np.delete(self.waypoints_monitor, passed_now_indices, axis=0)

        passed = bool(len(passed_now_indices))
        closest_waypoint_distance = np.min(distances)

        return closest_waypoint_distance, passed

    def reset(self):
        self.waypoints_monitor = np.copy(self.waypoints_monitor_for_reset)
        self.num_waypoints_passed = 0





