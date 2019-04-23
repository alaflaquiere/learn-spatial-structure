import numpy as np
import matplotlib.pyplot as plt

# TODO: rewrite the description
"""
The Environment is used to define the sensation associated with each position of the sensor.
"""


class GridWorld:
    """
    TODO
    """

    def __init__(self):
        self.type = "GridWorld"
        self.tore = True
        self.n_sensations = 4
        self.grid_size = (10, 10)
        self.pos2value_mapping = self.create_random_pos2sensation_mapping()

    def create_random_pos2sensation_mapping(self):
        """
        TODO
        """

        # scan all possible positions
        coordinates = np.meshgrid(np.arange(0, 1, 1/self.grid_size[0]), np.arange(0, 1, 1/self.grid_size[1]))

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

    def get_sensation_at_position(self, position):
        """
        Returns the sensations at a given set of input positions.
        (Warping is applied to the grid if self.tore=True.)

        Inputs:
            position - (N, 2) array

        Returns:
            sensations - (N, 4) array
        """

        if self.tore:  # warp the grid

            position[:, 0] = position[:, 0] % self.grid_size[0]
            position[:, 1] = position[:, 1] % self.grid_size[1]

            sensations = self.pos2value_mapping[position[:, 0], position[:, 1]]

        else:  # returns np.nan sensations for positions outside the grid

            valid_index = (position[:, 0] >= 0) & (position[:, 0] < self.grid_size[0]) & (position[:, 1] >= 0) & (position[:, 1] < self.grid_size[1])

            sensations = np.full((position.shape[0], self.n_sensations), np.nan)
            sensations[valid_index, :] = self.pos2value_mapping[position[valid_index, 0], position[valid_index, 1]]

        return sensations

    def generate_shift(self, k=1):
        """
        Returns k random shifts for the environment in [-5, 4]².
        """
        shift = np.array([np.random.randint(-self.grid_size[0], self.grid_size[0], k), np.random.randint(-self.grid_size[1], self.grid_size[1], k)])
        return shift.transpose()

    def display(self, ax=None):
        """
        Independently display the components of the sensations in the grid.

        ax - axe where to draw the surfaces
        """

        if ax is None:
            fig = plt.figure()
            ax = fig.gca(projection='3d')

        xx, yy = np.meshgrid(np.arange(self.grid_size[0]), np.arange(self.grid_size[1]))

        for i in range(self.n_sensations):
            ax.plot_surface(xx, yy, self.pos2value_mapping[:, :, i])
