import numpy as np
import matplotlib.pyplot as plt
import json
from flatland.env import Env

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
        the four sensory components is the composition of 3 random cosinus varying along x and 3 random cosinus varying along y.

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

        if self.tore:  # warp the grid

            position[:, 0] = position[:, 0] % self.env_size[0]
            position[:, 1] = position[:, 1] % self.env_size[1]

            sensations = self.pos2value_mapping[position[:, 0], position[:, 1]]

        else:  # returns np.nan sensations for positions outside the grid

            valid_index = (position[:, 0] >= 0) & (position[:, 0] < self.env_size[0]) & (position[:, 1] >= 0) & (position[:, 1] < self.env_size[1])

            sensations = np.full((position.shape[0], self.n_sensations), np.nan)
            sensations[valid_index, :] = self.pos2value_mapping[position[valid_index, 0], position[valid_index, 1]]

        if display:
            # todo: display it live
            pass

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


class Room:
    """
    todo
    """

    def __init__(self):
        """
        todo
        """
        # todo: move the display option to the get_sensation_at_position() method

        self.type = "Room"
        self.n_sensations = 10
        self.env_size = (150, 150)

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
        for i in range(15):

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
        todo
        """

        # deal with the case of a single position
        position = position.reshape(-1, 2)

        # prepare variable
        sensations = np.full((position.shape[0], self.n_sensations), np.nan)

        # define a null action
        actions = {'longitudinal_velocity': 0, 'lateral_velocity': 0, 'angular_velocity': 0, 'head_angle': 0}

        for i in range(position.shape[0]):

            if i % max(1, position.shape[0] // 100) == 0:
                print("\rsampling {} positions: {:3.0f}%".format(position.shape[0], (i + 1) / position.shape[0] * 100), end="")

            # set the agent position
            self.flatland_env.agent.body.position = (position[i, 0], position[i, 1])

            # get the sensation at that position
            # TODO: DISPLAY SENSOR if display=True
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

        # deal with the case of a single position
        position = [75, 75]

        # define a null action
        actions = {'longitudinal_velocity': 0, 'lateral_velocity': 0, 'angular_velocity': 0, 'head_angle': 0}

        # set the agent position
        self.flatland_env.agent.body.position = (position[0], position[1])

        # get the sensation at that position
        _, _, _, _ = self.flatland_env.step(actions, disp=True)

    def log(self, dest_log):
        """
        Writes the environment's attributes to the disk.
        """
        # todo, not functional yet
        pass
