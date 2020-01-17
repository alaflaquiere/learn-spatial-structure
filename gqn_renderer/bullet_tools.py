import os
import numpy as np
import pybullet
import colorsys
from gqn_renderer.bullet.camera import *
import sys

FLOOR_TEXTURES = [
    "textures/lg_floor_d.tga",
    "textures/lg_style_01_floor_blue_d.tga",
    "textures/lg_style_01_floor_orange_bright_d.tga"
]

WALL_TEXTURES = [
    "textures/lg_style_01_wall_cerise_d.tga",
    "textures/lg_style_01_wall_green_bright_d.tga",
    "textures/lg_style_01_wall_red_bright_d.tga",
    "textures/lg_style_02_wall_yellow_d.tga",
    "textures/lg_style_03_wall_orange_bright_d.tga"
]

# TODO: missing Icosahedron
OBJECTS = [
    pybullet.GEOM_CAPSULE,
    pybullet.GEOM_CYLINDER,
    # pybullet.GEOM_BOX,
    pybullet.GEOM_SPHERE
]

def get_colors(num_colors=6):
    colors = []
    for n in range(num_colors):
        hue = n / num_colors
        saturation = 1
        lightness = 1
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, lightness)
        colors.append([red, green, blue, 1])
    return colors


def build_scene(
        floor_textures=FLOOR_TEXTURES,
        wall_textures=WALL_TEXTURES,
        fix_light_position=False,
        gui=False):
    """
    Builds the scene
    """
    if gui:
        physics_client = pybullet.connect(pybullet.GUI)
        pybullet.configureDebugVisualizer(
            pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW,
            0,
            physicsClientId=physics_client)
        pybullet.configureDebugVisualizer(
            pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
            0,
            physicsClientId=physics_client)
        pybullet.configureDebugVisualizer(
            pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
            0,
            physicsClientId=physics_client)
    else:
        physics_client = pybullet.connect(pybullet.DIRECT)

    # Add current folder path to the bullet research path
    pybullet.setAdditionalSearchPath(
        os.path.dirname(os.path.realpath(__file__)))

    # Create the visual shapes for the floor and the walls
    floor_visual = pybullet.createVisualShape(
        shapeType=pybullet.GEOM_MESH,
        fileName="objects/floor.obj")
    wall_visual = pybullet.createVisualShape(
        shapeType=pybullet.GEOM_MESH,
        fileName="objects/wall.obj")

    # Create the bodies for the floor and the walls
    floor_body = pybullet.createMultiBody(
        baseMass=1,
        baseInertialFramePosition=[0, 0, 0],
        baseVisualShapeIndex=floor_visual,
        basePosition=[0.0, 0.0, 0.0],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
        useMaximalCoordinates=True)

    wall_bodies = list()
    wall_bodies.append(pybullet.createMultiBody(
        baseMass=1,
        baseInertialFramePosition=[0, 0, 0],
        baseVisualShapeIndex=wall_visual,
        basePosition=[0, -3.5, 1.15],
        baseOrientation=pybullet.getQuaternionFromEuler([-np.pi/2, 0, 0]),
        useMaximalCoordinates=True))
    wall_bodies.append(pybullet.createMultiBody(
        baseMass=1,
        baseInertialFramePosition=[0, 0, 0],
        baseVisualShapeIndex=wall_visual,
        basePosition=[0, 3.5, 1.15],
        baseOrientation=pybullet.getQuaternionFromEuler([-np.pi/2, 0, np.pi]),
        useMaximalCoordinates=True))
    wall_bodies.append(pybullet.createMultiBody(
        baseMass=1,
        baseInertialFramePosition=[0, 0, 0],
        baseVisualShapeIndex=wall_visual,
        basePosition=[3.5, 0, 1.15],
        baseOrientation=pybullet.getQuaternionFromEuler([-np.pi/2, 0, np.pi/2]),
        useMaximalCoordinates=True))
    wall_bodies.append(pybullet.createMultiBody(
        baseMass=1,
        baseInertialFramePosition=[0, 0, 0],
        baseVisualShapeIndex=wall_visual,
        basePosition=[-3.5, 0, 1.15],
        baseOrientation=pybullet.getQuaternionFromEuler([-np.pi/2, 0, -np.pi/2]),
        useMaximalCoordinates=True))

    # Update the floor and the wall's textures
    floor_texture = pybullet.loadTexture(np.random.choice(floor_textures))
    wall_texture = pybullet.loadTexture(np.random.choice(wall_textures))
    pybullet.changeVisualShape(floor_body, -1, textureUniqueId=floor_texture)

    for wall_body in wall_bodies:
        pybullet.changeVisualShape(wall_body, -1, textureUniqueId=wall_texture)

    # Light position
    if fix_light_position:
        translation = [1, -1, 1.6]
    else:
        xz = np.random.uniform(-1, 1, size=2)
        translation = np.array([xz[0], 1, xz[1]])

    pybullet.configureDebugVisualizer(
        lightPosition=translation,
        physicsClientId=physics_client)


# TODO: define the spawnble objects
def place_objects(
        colors,
        objects=OBJECTS,
        max_num_objects=3,
        min_num_objects=1,
        discrete_position=False,
        rotate_object=False):
    # directions = [-3., -1., 1., 3.]  # [-1.5, 0.0, 1.5]
    directions = [-2.5, -1, 1, 2.5]
    available_positions = []

    for y in directions:
        for x in directions:
            available_positions.append((x, y))

    available_positions = np.array(available_positions)
    num_objects = np.random.choice(range(min_num_objects, max_num_objects + 1))
    indices = np.random.choice(
        np.arange(len(available_positions)),
        replace=False,
        size=num_objects)

    for xy in available_positions[indices]:
        # TODO: Box shape fails
        object_visual = pybullet.createVisualShape(
            shapeType=np.random.choice(objects),
            rgbaColor=colors[np.random.choice(len(colors))])

        if not discrete_position:
            xy += np.random.uniform(-0.3, 0.3, size=xy.shape)
        if rotate_object:
            yaw = np.random.uniform(0, np.pi * 2, size=1)[0]
            # rotation = pyrender.quaternion.from_yaw(yaw)
            rotation = pybullet.getQuaternionFromEuler(0, -yaw, 0)
        else:
            rotation = [0, 0, 0, 1]

        pybullet.createMultiBody(
            baseMass=1,
            baseInertialFramePosition=[0, 0, 0],
            baseVisualShapeIndex=object_visual,
            basePosition=[xy[0], xy[1], 0],
            baseOrientation=rotation,
            useMaximalCoordinates=True)


def compute_yaw_and_pitch(vec):
    x, y, z = vec
    norm = np.linalg.norm(vec)
    if z < 0:
        yaw = np.pi + np.arctan(x / z)
    elif x < 0:
        if z == 0:
            yaw = np.pi * 1.5
        else:
            yaw = np.pi * 2 + np.arctan(x / z)
    elif z == 0:
        yaw = np.pi / 2
    else:
        yaw = np.arctan(x / z)
    pitch = -np.arcsin(y / norm)
    # return yaw, pitch
    return yaw, -pitch
    # return pitch, -yaw


def opengl_to_bullet_frame(vec):
    """
    Converts a rotation or a translation from the opengl ref frame to the
    bullet ref frame (Y-UP vs Z-UP)

    Parameters:
        vec - A vector xyz, list of 3 elements
    """
    return [vec[0], vec[2], -vec[1]]


def tear_down_scene():
    """
    Tears the scene down
    """
    pybullet.disconnect()


def main():
    build_scene()
    place_objects(
        get_colors(),
        max_num_objects=16,
        min_num_objects=16)

    camera = Camera(np.pi / 4, CameraResolution(16, 16))
    camera.setTranslation([0.2, 0.3, 0.4])
    frame = camera.getFrame()
    cv2.imshow("frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    tear_down_scene()


if __name__ == "__main__":
    main()
