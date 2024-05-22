import cv2
import numpy as np

# Load the image
image = cv2.imread('.\Line-Following.png')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

track_contour = contours[0]

track_points = [(point[0][0], point[0][1]) for point in track_contour[0::10]]

image_height, image_width = binary_image.shape

plane_width = 20.0
plane_height = 20.0

def pixel_to_world(pixel_x, pixel_y, image_width, image_height, plane_width, plane_height):
    world_x = (pixel_x / image_width) * plane_width - plane_width/2
    world_y = (pixel_y / image_height) * plane_height - plane_height/2
    return (world_x, world_y)


track_waypoints = [pixel_to_world(x, y, image_width, image_height, plane_width, plane_height) for x, y in track_points]

# for waypoint in track_waypoints:
#     print(waypoint)

np.save('track_waypoints', np.stack(track_waypoints))




