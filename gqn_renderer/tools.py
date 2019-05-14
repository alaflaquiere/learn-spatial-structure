import colorsys
import numpy as np
import gqn_renderer.pyrender as pyrender
from gqn_renderer.pyrender import (DirectionalLight, Mesh, Scene, Primitive, Node)
import trimesh
from PIL import Image
from OpenGL.GL import GL_LINEAR_MIPMAP_LINEAR


FLOOR_TEXTURES = [
    "gqn_renderer/textures/lg_floor_d.tga",
    "gqn_renderer/textures/lg_style_01_floor_blue_d.tga",
    "gqn_renderer/textures/lg_style_01_floor_orange_bright_d.tga"
]

WALL_TEXTURES = [
    "gqn_renderer/textures/lg_style_01_wall_cerise_d.tga",
    "gqn_renderer/textures/lg_style_01_wall_green_bright_d.tga",
    "gqn_renderer/textures/lg_style_01_wall_red_bright_d.tga",
    "gqn_renderer/textures/lg_style_02_wall_yellow_d.tga",
    "gqn_renderer/textures/lg_style_03_wall_orange_bright_d.tga",
]

OBJECTS = [
    pyrender.objects.Capsule,
    pyrender.objects.Cylinder,
    pyrender.objects.Icosahedron,
    pyrender.objects.Box,
    pyrender.objects.Sphere,
]


def get_colors(num_colors=6):
    colors = []
    for n in range(num_colors):
        hue = n / num_colors
        saturation = 1
        lightness = 1
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, lightness)
        colors.append(np.array((red, green, blue, 1)))
    return colors


def set_random_texture(node, path):
    texture_image = Image.open(path).convert("RGB")
    primitive = node.mesh.primitives[0]
    assert isinstance(primitive, Primitive)
    primitive.material.baseColorTexture.source = texture_image
    primitive.material.baseColorTexture.sampler.minFilter = GL_LINEAR_MIPMAP_LINEAR


def build_scene(floor_textures=FLOOR_TEXTURES, wall_textures=WALL_TEXTURES, fix_light_position=False):
    scene = Scene(
        bg_color=np.array([153 / 255, 226 / 255, 249 / 255]),
        ambient_light=np.array([0.5, 0.5, 0.5, 1.0]))

    floor_trimesh = trimesh.load("gqn_renderer/objects/floor.obj")
    mesh = Mesh.from_trimesh(floor_trimesh, smooth=False)
    node = Node(
        mesh=mesh,
        rotation=pyrender.quaternion.from_pitch(-np.pi / 2),
        translation=np.array([0, 0, 0]))
    texture_path = np.random.choice(floor_textures)
    set_random_texture(node, texture_path)
    scene.add_node(node)

    texture_path = np.random.choice(wall_textures)

    wall_trimesh = trimesh.load("gqn_renderer/objects/wall.obj")
    mesh = Mesh.from_trimesh(wall_trimesh, smooth=False)
    node = Node(mesh=mesh, translation=np.array([0, 1.15, -3.5]))
    set_random_texture(node, texture_path)
    scene.add_node(node)

    mesh = Mesh.from_trimesh(wall_trimesh, smooth=False)
    node = Node(
        mesh=mesh,
        rotation=pyrender.quaternion.from_yaw(np.pi),
        translation=np.array([0, 1.15, 3.5]))
    set_random_texture(node, texture_path)
    scene.add_node(node)

    mesh = Mesh.from_trimesh(wall_trimesh, smooth=False)
    node = Node(
        mesh=mesh,
        rotation=pyrender.quaternion.from_yaw(-np.pi / 2),
        translation=np.array([3.5, 1.15, 0]))
    set_random_texture(node, texture_path)
    scene.add_node(node)

    mesh = Mesh.from_trimesh(wall_trimesh, smooth=False)
    node = Node(
        mesh=mesh,
        rotation=pyrender.quaternion.from_yaw(np.pi / 2),
        translation=np.array([-3.5, 1.15, 0]))
    set_random_texture(node, texture_path)
    scene.add_node(node)

    light = DirectionalLight(color=np.ones(3), intensity=10)
    if fix_light_position:
        translation = np.array([1, 1, 1])
    else:
        xz = np.random.uniform(-1, 1, size=2)
        translation = np.array([xz[0], 1, xz[1]])
    yaw, pitch = compute_yaw_and_pitch(translation)
    node = Node(
        light=light,
        rotation=generate_camera_quaternion(yaw, pitch),
        translation=translation)
    scene.add_node(node)

    return scene


def place_objects(scene,
                  colors,
                  objects=OBJECTS,
                  max_num_objects=3,
                  min_num_objects=1,
                  discrete_position=False,
                  rotate_object=False):
    # Place objects
    directions = [-3., -1., 1., 3.]  # [-1.5, 0.0, 1.5]
    available_positions = []
    for z in directions:
        for x in directions:
            available_positions.append((x, z))
    available_positions = np.array(available_positions)
    num_objects = np.random.choice(range(min_num_objects, max_num_objects + 1))
    indices = np.random.choice(np.arange(len(available_positions)), replace=False, size=num_objects)
    for xz in available_positions[indices]:
        node = np.random.choice(objects)()
        node.mesh.primitives[0].color_0 = colors[np.random.choice(len(colors))]
        if not discrete_position:
            xz += np.random.uniform(-0.3, 0.3, size=xz.shape)
        if rotate_object:
            yaw = np.random.uniform(0, np.pi * 2, size=1)[0]
            rotation = pyrender.quaternion.from_yaw(yaw)
            parent = Node(
                children=[node],
                rotation=rotation,
                translation=np.array([xz[0], 0, xz[1]]))
        else:
            parent = Node(
                children=[node], translation=np.array([xz[0], 0, xz[1]]))
        scene.add_node(parent)


def udpate_vertex_buffer(cube_nodes):
    for node in (cube_nodes):
        node.mesh.primitives[0].update_vertex_buffer_data()


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
    return yaw, pitch


def generate_camera_quaternion(yaw, pitch):
    quaternion_yaw = pyrender.quaternion.from_yaw(yaw)
    quaternion_pitch = pyrender.quaternion.from_pitch(pitch)
    quaternion = pyrender.quaternion.multiply(quaternion_pitch, quaternion_yaw)
    quaternion = quaternion / np.linalg.norm(quaternion)
    return quaternion