import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from flatland.env import Env
import gqn_renderer as gqn
import gqn_renderer.tools as tools

"""
Collection of environments that can be used by generate-sensorimotor-data.py.
Environments are used to generate a sensory input for each position of the sensor (the environment thus implicitly includes information about the 
agent's sensor).

Each environment has the following attributes and methods:

    type: str
        type of environment
    n_sensations: int
        dimension of the sensory input produced at each sensor position
    env_size: (int/float, int/float)
        height and width of the environment

    get_sensation_at_position(position):
        generate the sensory inputs associated with input holistic sensor positions
    generate_shift(k):
        returns k random shifts of the environment
    display():
        displays the environment
    log(dir):
        logs the environment's parameters
"""


class GridWorld:
    """
    Tore gridworld of size (10, 10) with a sensory input of dimension 4 at each position of the grid.
    Each sensory component is generating using a random smooth periodic function of period 10 in both x and y in the grid.

    Attributes
    ----------
    type : str
        type of the environment
    tore : bool
        make the grid behave like a tore or not
    n_sensations : int
        number of independent sensory components
    env_size : list/tuple of 2 values
        size of the environment
    pos2value_mapping : ndarray
        mapping from position to sensations
    """

    def __init__(self):
        self.type = "GridWorld"
        self.tore = True
        self.n_sensations = 4
        self.env_size = (10, 10)
        self.pos2value_mapping = self.create_random_pos2sensation_mapping()

    def create_random_pos2sensation_mapping(self):
        """
        Generates a random smooth periodic function of period 10 in both x and y in the grid. Each function associated with one of
        the four sensory components is the composition of 3 random cosine varying along x and 3 random cosine varying along y.

        Return a mapping as an array of size (n_positions_in_the_grid, 4).
        """

        # scan all possible positions
        coordinates = np.meshgrid(np.arange(0, 1, 1/self.env_size[0]), np.arange(0, 1, 1/self.env_size[1]))

        # create the pos2sensation_mapping
        pos2sensation_mapping = np.full((len(coordinates[0][0]), len(coordinates[0][1]), self.n_sensations), np.nan)
        for i in range(self.n_sensations):

            # draw random parameters
            params = 4 * np.random.rand(12) - 2

            # generate the i-th sensation for all positions
            pos2sensation_mapping[:, :, i] \
                = 1 / params[0] * np.cos(2 * np.pi * (np.round(params[0]) * coordinates[0] + params[1])) \
                + 1 / params[2] * np.cos(2 * np.pi * (np.round(params[2]) * coordinates[1] + params[3])) \
                + 1 / params[4] * np.cos(2 * np.pi * (np.round(params[4]) * coordinates[0] + params[5])) \
                + 1 / params[6] * np.cos(2 * np.pi * (np.round(params[6]) * coordinates[1] + params[7])) \
                + 1 / params[8] * np.cos(2 * np.pi * (np.round(params[8]) * coordinates[0] + params[9])) \
                + 1 / params[10] * np.cos(2 * np.pi * (np.round(params[10]) * coordinates[1] + params[11]))

        return pos2sensation_mapping

    def get_sensation_at_position(self, position, display=False):
        """
        Returns the sensations at a given set of input positions.
        (Warping is applied to the grid if self.tore=True.)

        Inputs:
            position - (N, 2) array

        Returns:
            sensations - (N, 4) array
        """

        # deal with the case of a single position
        position = position.reshape(-1, 2)

        if self.tore:  # warp the grid

            position[:, 0] = position[:, 0] % self.env_size[0]
            position[:, 1] = position[:, 1] % self.env_size[1]

            sensations = self.pos2value_mapping[position[:, 0], position[:, 1]]

        else:  # returns np.nan sensations for positions outside the grid

            valid_index = (position[:, 0] >= 0) & (position[:, 0] < self.env_size[0]) & (position[:, 1] >= 0) & (position[:, 1] < self.env_size[1])

            sensations = np.full((position.shape[0], self.n_sensations), np.nan)
            sensations[valid_index, :] = self.pos2value_mapping[position[valid_index, 0], position[valid_index, 1]]

        if display:
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(1, 1, 1)
            for i in tqdm(range(position.shape[0]), desc="GridWorld", mininterval=1):
                ax.cla()
                ax.imshow(sensations[[i], :], interpolation="none")
                plt.pause(1e-8)
            plt.close(fig)

        return sensations

    def generate_shift(self, k=1, static=False):
        """
        Returns k random shifts for the environment in [-5, 4]².
        if static=True, returns the default shift which is self.env_size/2.
        """
        if static:
            shift = (np.array(self.env_size)//2) * np.ones((k, 2), dtype=int)
        else:
            shift = np.hstack((np.random.randint(-self.env_size[0], self.env_size[0], (k, 1)),
                               np.random.randint(-self.env_size[1], self.env_size[1], (k, 1))))
        return shift

    def display(self, ax=None):
        """
        Independently display the components of the sensations in the grid.

        ax - axe where to draw the surfaces
        """

        if ax is None:
            fig = plt.figure()
            ax = fig.gca(projection='3d')

        xx, yy = np.meshgrid(np.arange(self.env_size[0]), np.arange(self.env_size[1]))

        for i in range(self.n_sensations):
            ax.plot_surface(xx, yy, self.pos2value_mapping[:, :, i])

    def log(self, dest_log):
        """
        Writes the environment's attributes to the disk.
        """

        serializable_dict = self.__dict__.copy()
        for key, value in serializable_dict.items():
            if type(value) is np.ndarray:
                serializable_dict[key] = value.tolist()

        with open(dest_log, "w") as file:
            json.dump(serializable_dict, file, indent=1)


class FlatRoom:
    """
    A 2D room of size (150,150) randomly filled with random obstacles. The position (0,0) corresponds to the bottom left of the room.
    At each position, the environment generates a sensory input corresponding to the reading of a distance sensor array with a fixed orientation.

    Attributes
    ----------
    type : str
        type of the environment
    n_sensations : int
        number of independent sensory components
    env_size : list/tuple of 2 values
        size of the environment
    n_obstacles : int
        number of obstacles in the environment
    agent_parameters : dict
        agent parameters for the simulation
    game_parameters : dict
        simulation parameters
    flatland_env : flatland environment
        instance of the simulation
    """

    def __init__(self, n_obstacles=15, n_sensations=10):

        self.type = "FlatRoom"
        self.n_sensations = n_sensations
        self.env_size = (150, 150)
        self.n_obstacles = n_obstacles

        self.agent_parameters = {
            'radius': 5,
            'speed': 2,
            'rotation_speed': np.pi/10,
            'living_penalty': 0,
            'position': (75, 75),
            'angle': 0,
            'sensors': [
                {
                   'nameSensor': 'proximity',
                   'typeSensor': 'proximity',
                   'fovResolution': self.n_sensations,
                   'fovRange': 150,
                   'fovAngle': np.pi/2,
                   'bodyAnchor': 'body',
                   'd_r': 0,
                   'd_theta': 0,
                   'd_relativeOrientation': 0,
                   'display': True,
                }

            ],
            'actions': ['forward', 'turn_left', 'turn_right', 'left', 'right', 'backward'],
            'measurements': ['x', 'y', 'theta', 'head'],
            'texture': {
                'type': 'color',
                'c': (200, 200, 200)
            },
            'normalize_measurements': False,
            'normalize_states': False,
            'normalize_rewards': False
            }

        self.game_parameters = {
            'display': True,
            'horizon': 1001,
            'shape': self.env_size,
            'mode': 'time',

            'poisons': {'number': 0},
            'fruits': {'number': 0},

            'obstacles': [],
            'walls_texture': {
                'type': 'random_normal',
                'm': (np.random.randint(10, 240), np.random.randint(10, 240), np.random.randint(10, 240)),
                'd': (5, 5, 5),
            },
            'agent': self.agent_parameters
        }

        # create objects
        for i in range(n_obstacles):

            if np.random.random() < 0.5:

                self.game_parameters['obstacles'].append(
                    {
                        'shape': 'circle',
                        'position': (np.random.randint(1, 150), np.random.randint(1, 150)),
                        'radius': np.random.randint(5, 10),
                        'texture': {
                            'type': 'random_normal',
                            'm': (np.random.randint(10, 240), np.random.randint(10, 240), np.random.randint(10, 240)),
                            'd': (np.random.randint(1, 5), np.random.randint(1, 5), np.random.randint(1, 5)),
                        }
                    })
            else:

                self.game_parameters['obstacles'].append(
                    {
                        'shape': 'rectangle',
                        'position': (np.random.randint(1, 100), np.random.randint(1, 100)),
                        'width': np.random.randint(10, 20),
                        'length': np.random.randint(10, 20),
                        'angle': 'random',
                        'texture': {
                            'type': 'random_normal',
                            'm': (np.random.randint(10, 240), np.random.randint(10, 240), np.random.randint(10, 240)),
                            'd': (np.random.randint(1, 5), np.random.randint(1, 5), np.random.randint(1, 5)),
                        }
                    })

        self.flatland_env = Env(**self.game_parameters)

    def get_sensation_at_position(self, position, display=False):
        """
        Returns the sensations at a given set of input positions.

        Inputs:
            position - (N, 2) array

        Returns:
            sensations - (self.n_sensations, 4) array
        """

        # deal with the case of a single position
        position = position.reshape(-1, 2)

        # prepare variable
        sensations = np.full((position.shape[0], self.n_sensations), np.nan)

        # define a null action
        actions = {'longitudinal_velocity': 0, 'lateral_velocity': 0, 'angular_velocity': 0, 'head_angle': 0}

        for i in tqdm(range(position.shape[0]), desc="FlatRoom", mininterval=1):

            # set the agent position
            self.flatland_env.agent.body.position = (position[i, 0], position[i, 1])

            # get the sensation at that position
            sens, _, _, _ = self.flatland_env.step(actions, display)

            # check if there's been a collision
            collision = any((list(self.flatland_env.agent.body.position) != position[i, :]))

            # get the sensation if no collision
            if collision is False:
                sensations[i, :] = sens["proximity"]

        print(" done.")

        return sensations

    def generate_shift(self, k=1, static=False):
        """
        Returns k random shifts for the environment in [37.5, 112.5]² = 75 + [-75/2, +75/2]².
        If static=True, returns the default shift which is np.array(self.env_size)/2.
        """
        if static:
            shift = (np.array(self.env_size) / 2) * np.ones((k, 2))
        else:
            shift = np.array(self.env_size)/2 * np.random.rand(k, 2) + np.array(self.env_size)/4
        return shift

    def display(self):
        """
        Display the simulated environment with the agent at the center.
        """

        position = [75, 75]  # the agent is put in the center

        # define a null action
        actions = {'longitudinal_velocity': 0, 'lateral_velocity': 0, 'angular_velocity': 0, 'head_angle': 0}

        # set the agent position
        self.flatland_env.agent.body.position = (position[0], position[1])

        # get the sensation at that position
        _, _, _, _ = self.flatland_env.step(actions, disp=True)

    def log(self, dest_log):
        """
        Writes some environment's attributes to the disk.
        """

        serializable_dict = self.__dict__.copy()
        for key, value in self.__dict__.items():

            # keep only the ints, tuples, lists, and ndarrays
            if type(value) not in (int, tuple, list, np.ndarray):
                del serializable_dict[key]
                continue
            # make the ndarrays serializable
            if type(value) is np.ndarray:
                serializable_dict[key] = value.tolist()

        with open(dest_log, "w") as file:
            json.dump(serializable_dict, file, indent=1)


class GQNRoom:
    # todo: mention the original code somewhere
    """
    A 3D room of size (7,7) randomly filled with random objects. The position (0,0) corresponds to the center of the room.
    At each position, the environment generates a sensory input corresponding to the reading of a RGB camera with a fixed orientation.

    Attributes
    ----------
    type : str
        type of the environment
    n_sensations : int
        number of independent sensory components
    env_size : list/tuple of 2 values
        size of the environment
    n_obstacles : int
        number of obstacles in the environment
    scene : simulated environment
        instance of the simulation
    """

    def __init__(self, n_obstacles=7):

        self.type = "3dRoom"
        self.n_sensations = 16 * 16 * 3
        self.env_size = (7, 7)
        self.n_obstacles = n_obstacles

        # generate basic colors
        colors = tools.get_colors()

        # create the environment
        self.scene = tools.build_scene(fix_light_position=True)

        # create the objects
        tools.place_objects(self.scene, colors,
                                min_num_objects=self.n_obstacles,
                                max_num_objects=self.n_obstacles,
                                discrete_position=False,
                                rotate_object=False)

    def get_sensation_at_position(self, position, display=False):
        """
        Returns the sensations at a given set of input positions.

        Inputs:
            position - (N, 2) array

        Returns:
            sensations - (self.n_sensations, 4) array
        """

        # deal with the case of a single position
        position = position.reshape(-1, 2)

        # prepare variable
        sensations = np.full((position.shape[0], self.n_sensations), np.nan)

        # create the camera
        perspective_camera = gqn.pyrender.PerspectiveCamera(yfov=np.pi / 4)
        perspective_camera_node = gqn.pyrender.Node(camera=perspective_camera, translation=np.array([0, 1, 1]))

        # create the renderer
        renderer = gqn.pyrender.OffscreenRenderer(viewport_width=16, viewport_height=16)

        if display:
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(1, 1, 1)

        for i in tqdm(range(position.shape[0]), desc="GQNRoom", mininterval=1):

            # add the camera in the environment
            self.scene.add_node(perspective_camera_node)

            # set the camera position
            camera_position = [position[i, 0], 1.2, position[i, 1]]
            perspective_camera_node.translation = camera_position

            # set the camera orientation
            camera_direction = np.array([5, 1.8, 0])
            yaw, pitch = tools.compute_yaw_and_pitch(camera_direction)
            perspective_camera_node.rotation = tools.generate_camera_quaternion(yaw, pitch)

            # render
            image = renderer.render(self.scene, flags=gqn.pyrender.RenderFlags.SHADOWS_DIRECTIONAL)[0]

            # save sensation
            sensations[i, :] = image.reshape(-1)

            # clean the axis and display the image
            if display:
                ax.cla()
                ax.imshow(image, interpolation="none")
                plt.pause(1e-8)

            # remove the camera
            self.scene.remove_node(perspective_camera_node)

        if display:
            plt.close(fig)

        return sensations

    def generate_shift(self, k=1, static=False):
        """
        Returns k random shifts for the environment in [-1.75, 1.75]^2.
        If static=True, returns the default shift which is [0, 0].
        """
        if static:
            shift = np.zeros((k, 2))
        else:
            shift = np.array(self.env_size)/2 * np.random.rand(k, 2) - np.array(self.env_size)/4
        return shift

    def display(self, position=[8, 4.1, 0]):

        # create the camera
        perspective_camera = gqn.pyrender.PerspectiveCamera(yfov=np.pi / 4)
        perspective_camera_node = gqn.pyrender.Node(camera=perspective_camera, translation=np.array([0, 1, 1]))

        # create the renderer
        renderer = gqn.pyrender.OffscreenRenderer(viewport_width=16, viewport_height=16)

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(1, 1, 1)
        plt.tight_layout()

        # add the camera in the environment
        self.scene.add_node(perspective_camera_node)

        # set the camera position
        camera_position = position
        perspective_camera_node.translation = camera_position

        # set the camera orientation
        camera_direction = np.array([5, 1.8, 0])
        yaw, pitch = tools.compute_yaw_and_pitch(camera_direction)
        perspective_camera_node.rotation = tools.generate_camera_quaternion(yaw, pitch)

        # render
        image = renderer.render(self.scene, flags=gqn.pyrender.RenderFlags.SHADOWS_DIRECTIONAL)[0]

        # remove the camera
        self.scene.remove_node(perspective_camera_node)

        # clean the axis and display the image
        ax.cla()
        ax.imshow(image, interpolation="none")
        plt.pause(1e-8)

    def log(self, dest_log):
        """
        Writes the environment's attributes to the disk.
        """

        serializable_dict = self.__dict__.copy()
        for key, value in self.__dict__.items():

            # keep only the ints, tuples, lists, and ndarrays
            if type(value) not in (str, int, tuple, list, np.ndarray):
                del serializable_dict[key]
                continue
            # make the ndarrays serializable
            if type(value) is np.ndarray:
                serializable_dict[key] = value.tolist()

        with open(dest_log, "w") as file:
            json.dump(serializable_dict, file, indent=1)
