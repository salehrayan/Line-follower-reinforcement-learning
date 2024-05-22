import pybullet as p
from time import sleep
import pybullet_data



p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)

angle = p.addUserDebugParameter('Steering', -0.8, 0.8, 0)
throttle = p.addUserDebugParameter('Throttle', 0, 20, 0)

car = p.loadURDF('simplecar.urdf', [0, 0, 0.1])
plane = p.loadURDF('simple_plane - Copy.urdf')
p.changeDynamics(plane, -1, lateralFriction=4)

texture_id = p.loadTexture(r'E:\github\Line_follower_test\PyBullet_test\Code\utils\internet robot\track_texture.png')
p.changeVisualShape(
    plane,
    -1,
    textureUniqueId=texture_id)
# sleep(3)


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

