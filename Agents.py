import numpy as np
import matplotlib.pyplot as plt
import json

# TODO: add a save() method to save the envs and agents to the disk

"""
Collection of agents that can be used by generate-sensorimotor-data.py.
Agents are used to generate motor configurations and get the corresponding egocentric position of the sensor.

Each agent has the following attributes and methods:

    type: str
        type of agent
    n_motors : int
        number of independent motor components
    size_regular_grid: TODO
        TODO
        
    generate_random_sampling(k):
        randomly explores the motor space and return the motor samples and corresponding sensor egocentric positions
    generate_regular_sampling():
        returns a regular sampling of the motor space and the corresponding sensor egocentric positions
    display(motor):
        displays the agent's configuration associated with an input motor command
    log(dir):
        logs the agent's parameters
"""


class GridExplorer:
    """
    A ``discrete'' agent that can move its sensor in a 5x5 grid using a non-linear redundant mapping from 3 motors to positions in the 2D grid.

    Attributes
    ----------
    type : str
        type of the agent
    n_motors : int
        number of independent motor components, each motor is in [-1, 1]
    n_states: int
        number of possible motor states
    state2motor_mapping : nd.array of float of size (n_states, n_motors)
        mapping between the states and the motor configurations
    state2pos_mapping : nd.array of int of size (n_states, 2)
        mapping between the states and the sensor position (x, y) in [-2, 2]² in the grid
    size_regular_grid : int
        resolution of the regular grid of motor configurations used for evaluation
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

    def display(self, motor):
        """
        Displays the position associated with a motor configuration
        """
        for i in range(motor.shape[0]):
            # get the state from the motor command
            state_index = [np.where(np.all(self.state2motor_mapping == motor[i, :], axis=1)) for i in range(len(motor))]
            position = self.state2pos_mapping[state_index, :]
            plt.plot(position[i, 0], position[i, 1], 'xk')

    def log(self, dest_log):
        """
        Writes the agent's attributes to the disk.
        """

        serializable_dict = self.__dict__.copy()
        for key, value in serializable_dict.items():
            if type(value) is np.ndarray:
                serializable_dict[key] = value.tolist()

        with open(dest_log, "w") as file:
            json.dump(serializable_dict, file, indent=1)


class ManyDofGridExplorer:
    """
    A ``discrete'' agent that can move its sensor in a 5x5 grid using a non-linear redundant mapping from many motors to positions in the 2D grid.

    Attributes
    ----------
    type : str
        type of the agent
    n_motors : int
        number of independent motor components, each motor is in [-1, 1]
    n_states: int
        number of possible motor states
    state2motor_mapping : nd.array of float of size (n_states, n_motors)
        mapping between the states and the motor configurations
    state2pos_mapping : nd.array of int of size (n_states, 2)
        mapping between the states and the sensor position (x, y) in [-2, 2]² in the grid
    size_regular_grid : int
        resolution of the regular grid of motor configurations used for evaluation
    """

    def __init__(self):
        self.type = "ManyDofGridExplorer"
        self.n_motors = 6
        self.n_states = 5**self.n_motors
        self.state2motor_mapping, self.state2pos_mapping = self.create_random_motor2pos_mapping()
        self.size_regular_grid = self.n_states

    def create_random_motor2pos_mapping(self):
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
        coordinates = np.array(np.meshgrid(*list([np.linspace(0, 1, 5)]) * self.n_motors))

        # reshape the coordinates into matrix of size (5**self.n_motors, self.n_motors)
        mapping = np.full((np.size(coordinates[0]), len(coordinates)), np.nan)
        for i in range(len(coordinates)):
            mapping[:, i] = coordinates[i].reshape((-1))

        # apply a transformation to the mapping to create the non-linear state2motor_mapping
        #
        ###### ORIGINAL ######
        # mixing matrix
        mixing_matrix = 4 * np.random.rand(self.n_motors, self.n_motors) - 2
        state2motor_mapping = np.matmul(mapping, np.linalg.inv(mixing_matrix))
        #
        # normalization of the values into [0, 1] for easy application of the non-linearities
        state2motor_mapping = state2motor_mapping - np.min(state2motor_mapping, axis=0)
        state2motor_mapping = state2motor_mapping / np.max(state2motor_mapping, axis=0)
        #
        # non-linear transformation from [0, 1] to [0, 1]
        state2motor_mapping[:, 0] = np.power(state2motor_mapping[:, 0], 0.5, dtype=float)
        state2motor_mapping[:, 1] = np.power(state2motor_mapping[:, 1], 2, dtype=float)
        state2motor_mapping[:, 2] = np.power(state2motor_mapping[:, 2], 3, dtype=float)
        state2motor_mapping[:, 3] = (np.log((state2motor_mapping[:, 3] + 0.1) / 1.1) - np.log(0.1 / 1.1)) / (-np.log(0.1 / 1.1))
        state2motor_mapping[:, 4] = (np.exp(state2motor_mapping[:, 4]) - 1) / (np.exp(1) - 1)
        state2motor_mapping[:, 5] = np.power(state2motor_mapping[:, 5], 1, dtype=float)
        ########################
        #
        ###### MODIFIED (not better reults) ######
        # # create a suitable mixing matrix
        # inverse_mixing_matrix = 4 * np.random.rand(self.n_motors, self.n_motors)
        # # normalize the inverse mixing matrix such that its columns have a norm of sqrt(self.n_motors)
        # # that way the states contained in the hypercube [0,1]^6 are projected into a skewed hypercube contained in [0,1]^6
        # # and the non-linear transformations can be applied without any issue
        # inverse_mixing_matrix = inverse_mixing_matrix / (np.linalg.norm(inverse_mixing_matrix, axis=0, keepdims=True) * np.sqrt(self.n_motors))
        # # get the associated forward mixing matrix (not required)
        # #mixing_matrix = np.linalg.inv(inverse_mixing_matrix)
        # #
        # state2motor_mapping = np.matmul(mapping, inverse_mixing_matrix)
        # #
        # # SANITY CHECK raise error of outside of [0,1]
        # if any(np.min(state2motor_mapping, axis=0) < 0) or any(np.max(state2motor_mapping, axis=0) > 1):
        #     print("[!] PROBLEM WITH THE MIXING MATRIX, some motor states are out of [0,1]")
        #     print(np.min(state2motor_mapping, axis=0))
        #     print(np.max(state2motor_mapping, axis=0))
        #     input("press a key to continue")
        # #
        # # non-linear transformation from [0, 1] to [0, 1]
        # state2motor_mapping[:, 0] = np.power(state2motor_mapping[:, 0], 0.5, dtype=float)  # inverse of np.power(?, 2)
        # state2motor_mapping[:, 1] = np.power(state2motor_mapping[:, 1], 2, dtype=float)    # inverse of np.sqrt(?)
        # state2motor_mapping[:, 2] = np.power(state2motor_mapping[:, 2], 3, dtype=float)    # inverse of np.cbrt(?)
        # state2motor_mapping[:, 3] = (np.log((state2motor_mapping[:, 3] + 0.1) / 1.1) - np.log(0.1 / 1.1)) / (-np.log(0.1 / 1.1))  # inverse of 0.1 * np.power(1.1 / 0.1, ?) - 0.1
        # state2motor_mapping[:, 4] = (np.exp(state2motor_mapping[:, 4]) - 1) / (np.exp(1) - 1)  # inverse of np.log(x * (np.exp(1) - 1) + 1)
        # state2motor_mapping[:, 5] = np.power(state2motor_mapping[:, 5], 1, dtype=float)  # inverse of identity function
        ########################

        # rescale the motor commands in the state2motor_mapping in [-1, 1]
        state2motor_mapping = state2motor_mapping * 2 - 1

        # apply a trivial transformation (drop the last coordinates) to the mapping to create the state2pos_mapping
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

    def display(self, motor):
        """
        Displays the position associated with a motor configuration
        """
        for i in range(motor.shape[0]):
            # get the state from the motor command
            state_index = [np.where(np.all(self.state2motor_mapping == motor[i, :], axis=1)) for i in range(len(motor))]
            position = self.state2pos_mapping[state_index, :]
            plt.plot(position[i, 0], position[i, 1], 'xk')

    def log(self, dest_log):
        """
        Writes the agent's attributes to the disk.
        """

        serializable_dict = self.__dict__.copy()
        for key, value in serializable_dict.items():
            if type(value) is np.ndarray:
                serializable_dict[key] = value.tolist()

        with open(dest_log, "w") as file:
            json.dump(serializable_dict, file, indent=1)


class HingeArm:
    """
    A three-segment arm with hinge joints that can move its end-effector in a 2D space.
    The arm has three segments of size 12 and covers a working space of radius 36.
    Note that only the position of the end-effector is considered; its orientation is not computed.

    Attributes
    ----------
    type : str
        type of the agent
    n_motors : int
        number of independent motor components; each motor is in [-1, 1], which maps to [-2*self.motor_amplitude, 2*self.motor_amplitude]
    motor_amplitude : float
        scale by which the motor components are multiplied
    segments_length : list/tuple of 3 float/int
        lengths of the arm segments
    size_regular_grid: int
        resolution of the regular grid of motor configurations used for evaluation
    """

    def __init__(self, segments_length=(12, 12, 12)):
        self.type = "HingeArm"
        self.n_motors = 3
        self.motor_amplitude = np.pi
        self.segments_length = segments_length  # the arm covers a working space of radius 36 in an environment of size size 150
        self.reso_regular_grid = 6
        self.size_regular_grid = self.reso_regular_grid ** self.n_motors

    def get_position_from_motor(self, motor):
        """
        Get the coordinates of the sensor via trigonometry.
        """
        x = np.sum(np.multiply(self.segments_length, np.cos(np.cumsum(self.motor_amplitude * motor, axis=1))), axis=1, keepdims=True)
        y = np.sum(np.multiply(self.segments_length, np.sin(np.cumsum(self.motor_amplitude * motor, axis=1))), axis=1, keepdims=True)
        return np.hstack((x, y))

    def generate_random_sampling(self, k=1):
        """
        Draw a set of k randomly selected motor configurations and associated egocentric sensor positions

        Returns:
            motor - (k, 3) array
            position - (k, 2) array
        """

        # draw random motor components in [-1,1]
        motor = 2 * np.random.rand(k, self.n_motors) - 1

        # get the associated egocentric positions in [-36, 36]
        position = self.get_position_from_motor(motor)

        return motor, position

    def generate_regular_sampling(self):
        """
        Generates a regular grid of motor configurations in the motor space.
        """

        xx, yy, zz = np.meshgrid(np.linspace(-1, 1, self.reso_regular_grid),
                                 np.linspace(-1, 1, self.reso_regular_grid),
                                 np.linspace(-1, 1, self.reso_regular_grid))
        motor_grid = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)))

        pos_grid = self.get_position_from_motor(motor_grid)

        return motor_grid, pos_grid

    def display(self, motor):
        """
        Displays the position associated with a motor configuration
        """

        # get the joints positions
        x = np.cumsum(np.multiply(self.segments_length, np.cos(np.cumsum(self.motor_amplitude * motor, axis=1))), axis=1, keepdims=True)
        y = np.cumsum(np.multiply(self.segments_length, np.sin(np.cumsum(self.motor_amplitude * motor, axis=1))), axis=1, keepdims=True)

        # add the agent's base
        x = np.hstack((np.zeros((x.shape[0], 1)), x))
        y = np.hstack((np.zeros((y.shape[0], 1)), y))

        # unify the cases where motor is a unique input or multiple inputs
        motor = motor.reshape((-1, self.n_motors))

        # display the different motor configurations
        for i in range(motor.shape[0]):
            plt.plot(x[i, :, 0], y[i, :, 1], '-ok')

    def log(self, dest_log):
        """
        Writes the agent's attributes to the disk.
        """

        serializable_dict = self.__dict__.copy()
        for key, value in serializable_dict.items():
            if type(value) is np.ndarray:
                serializable_dict[key] = value.tolist()

        with open(dest_log, "w") as file:
            json.dump(serializable_dict, file, indent=1)


class ManyDofHingeArm:
    """
    A four-segment arm with hinge and translational joints that can move its end-effector in a 2D space.
    The arm has four segments of size 9 and covers a working space of radius 36.
    Note that only the position of the end-effector is considered; its orientation is not computed.

    Attributes
    ----------
    type : str
        type of the agent
    n_motors : int
        number of independent motor components; each motor is in [-1, 1], which maps to [-2*self.motor_amplitude, 2*self.motor_amplitude]
    motor_amplitude : list of float
        scale by which the motor components are multiplied
    segments_length : list/tuple of 3 float/int
        lengths of the arm segments
    size_regular_grid: int
        resolution of the regular grid of motor configurations used for evaluation
    """

    def __init__(self, segments_length=(9, 9, 9, 9)):
        self.type = "ManyDofHingeArm"
        self.n_motors = 6
        self.motor_amplitude = [np.pi, np.pi, np.pi, np.pi, 1, 1]
        self.segments_length = segments_length  # the arm covers a working space of radius 36 in an environment of size size 150
        self.reso_regular_grid = 5
        self.size_regular_grid = self.reso_regular_grid**self.n_motors

    def get_position_from_motor(self, motor):
        """
        Get the coordinates of the sensor via trigonometry.
        """
        seg_lengths = np.array(self.segments_length) * np.hstack((np.ones((motor.shape[0], 1)), self.motor_amplitude[4:] * motor[:, 4:], np.ones((motor.shape[0], 1))))
        x = np.sum(np.multiply(seg_lengths, np.cos(np.cumsum(self.motor_amplitude[0:4] * motor[:, 0:4], axis=1))), axis=1, keepdims=True)
        y = np.sum(np.multiply(seg_lengths, np.sin(np.cumsum(self.motor_amplitude[0:4] * motor[:, 0:4], axis=1))), axis=1, keepdims=True)
        return np.hstack((x, y))

    def generate_random_sampling(self, k=1):
        """
        Draw a set of k randomly selected motor configurations and associated egocentric sensor positions

        Returns:
            motor - (k, 6) array
            position - (k, 2) array
        """

        # draw random motor components in [-1,1]
        motor = 2 * np.random.rand(k, self.n_motors) - 1

        # get the associated egocentric positions in [-36, 36]
        position = self.get_position_from_motor(motor)

        return motor, position

    def generate_regular_sampling(self):
        """
        Generates a regular grid of motor configurations in the motor space.
        """

        xx, yy, zz, aa, bb, cc = np.meshgrid(np.linspace(-1, 1, self.reso_regular_grid),
                                             np.linspace(-1, 1, self.reso_regular_grid),
                                             np.linspace(-1, 1, self.reso_regular_grid),
                                             np.linspace(-1, 1, self.reso_regular_grid),
                                             np.linspace(-1, 1, self.reso_regular_grid),
                                             np.linspace(-1, 1, self.reso_regular_grid))
        motor_grid = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1), aa.reshape(-1, 1), bb.reshape(-1, 1), cc.reshape(-1, 1)))

        pos_grid = self.get_position_from_motor(motor_grid)

        return motor_grid, pos_grid

    def display(self, motor):
        """
        Displays the position associated with a motor configuration
        """

        # get the joints positions
        seg_lengths = np.array(self.segments_length) * np.hstack((np.ones((motor.shape[0], 1)), self.motor_amplitude[4:] * motor[:, 4:], np.ones((motor.shape[0], 1))))
        x = np.cumsum(np.multiply(seg_lengths, np.cos(np.cumsum(self.motor_amplitude[0:4] * motor[:, 0:4], axis=1))), axis=1)
        y = np.cumsum(np.multiply(seg_lengths, np.sin(np.cumsum(self.motor_amplitude[0:4] * motor[:, 0:4], axis=1))), axis=1)

        # add the agent's base
        x = np.hstack((np.zeros((x.shape[0], 1)), x))
        y = np.hstack((np.zeros((y.shape[0], 1)), y))

        # unify the cases where motor is a unique input or multiple inputs
        motor = motor.reshape((-1, self.n_motors))

        # display the different motor configurations
        for i in range(motor.shape[0]):
            plt.plot(x[i, :], y[i, :], '-ok')

    def log(self, dest_log):
        """
        Writes the agent's attributes to the disk.
        """

        serializable_dict = self.__dict__.copy()
        for key, value in serializable_dict.items():
            if type(value) is np.ndarray:
                serializable_dict[key] = value.tolist()

        with open(dest_log, "w") as file:
            json.dump(serializable_dict, file, indent=1)
