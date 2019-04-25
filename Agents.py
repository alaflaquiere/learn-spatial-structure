import numpy as np
import matplotlib.pyplot as plt


# todo: rewrite the description
"""
All agents have the following attributes and methods:
- type
- number_motors
- generate_motor_and_get_position()
- generate_regular_sampling()
- save_log()

The agent is only used to generated motor configurations and their corresponding egocentric sensor positions.
"""


class GridExplorer:
    """
    A discrete agent that can move its sensor in a 5x5 grid using a non-linear redundant mapping from 3 motors to positions n the 2D grid.

    Attributes
    ----------
    type: str
        type of the agent
    n_motors : int
        number of independent motor components, each motor is in [-1, 1]
    n_states : int
        number of possible motor states
    state2motor_mapping : nd.array of float of size (n_states, n_motors)
        mapping between the states and the motor configurations
    state2pos_mapping : nd.array of int of size (n_states, 2)
        mapping between the states and the sensor position (x, y) in [-2, 2]² in the grid
    """

    def __init__(self):
        self.type = "GridExplorer"
        self.n_motors = 3
        self.n_states = 5*5*5
        self.state2motor_mapping, self.state2pos_mapping = self.create_random_motor2pos_mapping()
        self.size_regular_grid = self.n_states

    @staticmethod
    def create_random_motor2pos_mapping():
        """
        Create a random mapping from motor configurations to egocentric positions of the sensor.
        The mapping is discrete, such that a limited set of motor configurations is associated with a limited set of sensors positions (of smaller
        cardinal due to redundancy). For convenience, the mapping is split into a state2motor_mapping and a state2pos_mapping, where a state
        corresponds to the index given to a motor configuration.

        Returns:
            state2motor_mapping - (n_states, 3) array
            state2pos_mapping - (n_states, 2) array
        """

        # scan all possible states
        coordinates = np.array(np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5), np.linspace(0, 1, 5)))

        # reshape the coordinates into matrix of size (5*5*5, 3)
        mapping = np.full((np.size(coordinates[0]), len(coordinates)), np.nan)
        for i in range(len(coordinates)):
            mapping[:, i] = coordinates[i].reshape((-1))

        # apply a transformation to the mapping to create the non-linear state2motor_mapping
        state2motor_mapping = np.power(mapping, 3, dtype=float)

        # rescale the motor commands in the state2motor_mapping in [-1, 1]
        state2motor_mapping = state2motor_mapping * 2 - 1

        # apply a trivial transformation (drop the last coordinate) to the mapping to create the state2pos_mapping
        state2pos_mapping = mapping[:, 0:2]

        # rescale the positions in [-2, 2] and format it as integer
        state2pos_mapping = ((state2pos_mapping - 0.5) * 4).astype(int)

        return state2motor_mapping, state2pos_mapping

    def generate_random_sampling(self, k=1):
        """
        Draw a set of k randomly selected motor configurations and associated sensor positions

        Returns:
            motor - (k, 3) array
            position - (k, 2) array
        """

        state_index = np.random.randint(0, self.n_states, k)

        motor = self.state2motor_mapping[state_index, :]
        position = self.state2pos_mapping[state_index, :]

        return motor, position

    def generate_regular_sampling(self):
        """
        Returns all the motor configurations and associated sensor positions.
        """
        return self.state2motor_mapping, self.state2pos_mapping

    def get_state_from_motor(self, motor):
        """
        Get the state associated with the input motor state
        """
        state = [np.where(np.all(self.state2motor_mapping == motor[i, :], axis=1)) for i in range(len(motor))]
        return state

    def display(self, motor):
        """
        Displays the position associated with a motor configuration
        """
        for i in range(motor.shape[0]):
            state_index = self.get_state_from_motor(motor[i, :])
            position = self.state2pos_mapping[state_index, :]
            plt.plot(position[i, 0], position[i, 1], 'xk')
