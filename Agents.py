import numpy as np
import matplotlib.pyplot as plt
import json
import _pickle as cpickle

"""
Collection of agents that can be used by generate-sensorimotor-data.py.
Agents are used to generate motor configurations and get the corresponding egocentric position of the sensor.
"""


class Agent:
    """
        type: str
            type of agent
        n_motors : int
            number of independent motor components in [-1, 1]
        size_regular_grid: int
            number of samples that form the regular sampling of the motor space

        generate_random_sampling(k):
            randomly explores the motor space and returns the motor samples and corresponding sensor egocentric positions
        generate_regular_sampling():
            returns a regular sampling of the motor space and the corresponding sensor egocentric positions
        display(motor):
            displays the agent's configuration associated with an input motor command
        log(dir):
            logs the agent's parameters
    """

    def __init__(self, type_agent, n_motors, size_regular_grid):
        self.type = type_agent
        self.n_motors = n_motors
        self.size_regular_grid = size_regular_grid

    def generate_random_sampling(self, k):
        return None, None

    def generate_regular_sampling(self):
        return None, None

    def display(self, motor_configuration):
        pass

    def save(self, destination):
        """
        Save the agent to the disk.
        """

        try:
            # save a readable log of the agent
            serializable_dict = self.__dict__.copy()
            for key, value in serializable_dict.items():
                if type(value) is np.ndarray:
                    serializable_dict[key] = value.tolist()
            with open(destination + "/agent_params.txt", "w") as f:
                json.dump(serializable_dict, f, indent=2, sort_keys=True)

            # save the object on disk
            with open(destination + "/agent.pkl", "wb") as f:
                cpickle.dump(self, f)

        except:
            print("ERROR: saving the agent in {} failed".format(destination))
            return False


class GridExplorer(Agent):
    """
        A ``discrete'' agent that can move its sensor in a 5x5 grid using a non-linear redundant mapping
        from motors configurations to positions in the 2D grid.
        Attributes
        ----------
        n_states: int
            number of possible motor states
        state2motor_mapping : nd.array of float of size (n_states, n_motors)
            mapping between the states and the motor configurations
        state2pos_mapping : nd.array of int of size (n_states, 2)
            mapping between the states and the sensor position (x, y) in [-2, 2]² in the grid
        """

    def __init__(self, type_agent, n_motors, size_regular_grid):
        super().__init__(type_agent, n_motors, size_regular_grid)
        self.state2motor_mapping = None
        self.state2pos_mapping = None
        self.n_states = size_regular_grid

    def generate_random_sampling(self, k=1):
        """
        Draw a set of k randomly selected motor configurations and associated sensor positions
        Returns:
            motor - (k, self.n_motors) array
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
        plt.plot(0, 0, 'ks')
        for i in range(motor.shape[0]):
            # get the state from the motor command
            state_index = np.where(np.all(self.state2motor_mapping == motor[i, :], axis=1))[0][0]
            position = self.state2pos_mapping[state_index, :]
            plt.plot(position[0], position[1], 'ro')


class GridExplorer3dof(GridExplorer):
    """
    A ``discrete'' agent that can move its sensor in a 5x5 grid using a non-linear redundant mapping from 3 motors to positions in the 2D grid.
    Attributes
    ----------
    state2motor_mapping : nd.array of float of size (n_states, n_motors)
        mapping between the states and the motor configurations
    state2pos_mapping : nd.array of int of size (n_states, 2)
        mapping between the states and the sensor position (x, y) in [-2, 2]² in the grid
    """

    def __init__(self, number_motors=3, resolution=5):
        super().__init__(type_agent="GridExplorer3dof", n_motors=number_motors, size_regular_grid=resolution**number_motors)
        self.state2motor_mapping, self.state2pos_mapping = self.create_discrete_mapping_3dmotor_to_2dpos(reso=resolution)

    def create_discrete_mapping_3dmotor_to_2dpos(self, reso):
        """
        Create a random mapping from motor configurations to egocentric positions of the sensor.
        The mapping is discrete, such that a limited set of motor configurations is associated with a limited set of sensors positions (of smaller
        cardinality due to redundancy). For convenience, the mapping is split into a state2motor_mapping and a state2pos_mapping, where a state
        corresponds to the index given to a motor configuration.

        Returns:
            state2motor_mapping - (reso**3, 3) array
            state2pos_mapping - (reso**3, 2) array
        """

        # scan all possible states
        coordinates = np.array(np.meshgrid(*list([np.linspace(0, 1, reso)]) * self.n_motors))

        # reshape the coordinates into matrix of size (reso**3, 3)
        mapping = np.array([coord.reshape((-1)) for coord in coordinates]).T

        # apply a transformation to the mapping to create the non-linear state2motor_mapping
        state2motor_mapping = np.power(mapping, 3, dtype=float)

        # rescale the motor commands in the state2motor_mapping in [-1, 1]
        state2motor_mapping = state2motor_mapping * 2 - 1

        # apply a trivial transformation (drop the last coordinate) to the mapping to create the redundant state2pos_mapping
        state2pos_mapping = mapping[:, 0:2]

        # rescale the positions in [-2, 2] and format it as integer
        state2pos_mapping = (state2pos_mapping * 4 - 2).astype(int)

        return state2motor_mapping, state2pos_mapping


class GridExplorer6dof(GridExplorer):
    """
    A ``discrete'' agent that can move its sensor in a 5x5 grid using a non-linear redundant mapping from 6 motors to positions in the 2D grid.
    Attributes
    ----------
    state2motor_mapping : nd.array of float of size (n_states, n_motors)
        mapping between the states and the motor configurations
    state2pos_mapping : nd.array of int of size (n_states, 2)
        mapping between the states and the sensor position (x, y) in [-2, 2]² in the grid
    """

    def __init__(self, number_motors=6, resolution=4):
        super().__init__(type_agent="GridExplorer3dof", n_motors=number_motors, size_regular_grid=resolution**number_motors)
        self.state2motor_mapping, self.state2pos_mapping = self.create_discrete_mapping_6dmotor_to_2dpos(reso=resolution)

    def create_discrete_mapping_6dmotor_to_2dpos(self, reso):
        """
        Create a random mapping from motor configurations to egocentric positions of the sensor.
        The mapping is discrete, such that a limited set of motor configurations is associated with a limited set of sensors positions (of smaller
        cardinality due to redundancy). For convenience, the mapping is split into a state2motor_mapping and a state2pos_mapping, where a state
        corresponds to the index given to a motor configuration.
        Returns:
            state2motor_mapping - (reso**n_motors, 6) array
            state2pos_mapping - (reso**n_motors, 2) array
        """

        # scan all possible states
        coordinates = np.array(np.meshgrid(*list([np.linspace(0, 1, reso)]) * self.n_motors))

        # reshape the coordinates into matrix of size (reso**n_motors, n_motors)
        mapping = np.array([coord.reshape((-1)) for coord in coordinates]).T

        # mixing matrix
        mixing_matrix = 4 * np.random.rand(self.n_motors, self.n_motors) - 2
        state2motor_mapping = np.matmul(mapping, np.linalg.inv(mixing_matrix))

        # normalization of the values into [0, 1] for easy application of the non-linearities
        state2motor_mapping = state2motor_mapping - np.min(state2motor_mapping, axis=0)
        state2motor_mapping = state2motor_mapping / np.max(state2motor_mapping, axis=0)

        # non-linear transformation from [0, 1] to [0, 1]
        state2motor_mapping[:, 0] = np.power(state2motor_mapping[:, 0], 0.5, dtype=float)
        state2motor_mapping[:, 1] = np.power(state2motor_mapping[:, 1], 2, dtype=float)
        state2motor_mapping[:, 2] = np.power(state2motor_mapping[:, 2], 3, dtype=float)
        state2motor_mapping[:, 3] = (np.log((state2motor_mapping[:, 3] + 0.1) / 1.1) - np.log(0.1 / 1.1)) / (-np.log(0.1 / 1.1))
        state2motor_mapping[:, 4] = (np.exp(state2motor_mapping[:, 4]) - 1) / (np.exp(1) - 1)
        state2motor_mapping[:, 5] = np.power(state2motor_mapping[:, 5], 1, dtype=float)

        # rescale the motor commands in the state2motor_mapping in [-1, 1]
        state2motor_mapping = state2motor_mapping * 2 - 1

        # apply a trivial transformation (drop the last coordinates) to the mapping to create the state2pos_mapping
        state2pos_mapping = mapping[:, 0:2]

        # rescale the positions in [-2, 2] and format it as integer
        state2pos_mapping = (state2pos_mapping * 4 - 2).astype(int)

        return state2motor_mapping, state2pos_mapping


class HingeArm(Agent):
    """
        An arm that can move its end-effector in a 2D space, while keeping its orientation fixed.
        Attributes
        ----------
        motor_amplitude : float
            scale by which the motor components are multiplied to command the joints
        segments_length : list/tuple of 3 float/int
            lengths of the arm segments
        """

    def __init__(self, type_agent, n_motors, size_regular_grid):
        super().__init__(type_agent, n_motors, size_regular_grid)
        self.motor_amplitude = None
        self.segments_length = None

    def get_position_from_motor(self, motor):
        return None

    def generate_random_sampling(self, k=1):
        """
        Draw a set of k randomly selected motor configurations and associated egocentric sensor positions
        Returns:
            motor - (k, self.n_motors) array
            position - (k, 2) array
        """
        # draw random motor components in [-1, 1]
        motor = 2 * np.random.rand(k, self.n_motors) - 1
        # get the associated egocentric positions
        position = self.get_position_from_motor(motor)
        return motor, position

    def generate_regular_sampling(self):
        """
        Generates a regular grid of motor configurations in the motor space.
        """
        resolution = int(self.size_regular_grid**(1/self.n_motors))
        # create a grid of coordinates
        coordinates = np.array(np.meshgrid(*list([np.arange(-1, 1, 2/resolution)]) * self.n_motors))
        # reshape the coordinates into matrix of size (reso**n_motors, n_motors)
        motor_grid = np.array([coord.reshape((-1)) for coord in coordinates]).T
        # get the corresponding positions
        pos_grid = self.get_position_from_motor(motor_grid)
        return motor_grid, pos_grid


class HingeArm3dof(HingeArm):
    """
    A three-segment arm with hinge joints that can move its end-effector in a 2D space.
    The arm has three segments of size 12 and covers a working space of radius 36.
    Note that only the position of the end-effector is considered; its orientation is considered fixed.
    Attributes
    ----------
    motor_amplitude : float
        scale by which the motor components are multiplied
    segments_length : list/tuple of 3 float/int
        lengths of the arm segments
    """

    def __init__(self):
        super().__init__(type_agent="HingeArm3dof", n_motors=3, size_regular_grid=7**3)
        self.motor_amplitude = [np.pi] * self.n_motors
        self.segments_length = [0.5] * self.n_motors  # the arm covers a working space working space of radius 1.5 in an environment of size size 7

    def get_position_from_motor(self, motor):
        """
        Get the coordinates of the sensor via trigonometry.
        """
        x = np.sum(np.multiply(self.segments_length, np.cos(np.cumsum(self.motor_amplitude * motor, axis=1))), axis=1, keepdims=True)
        y = np.sum(np.multiply(self.segments_length, np.sin(np.cumsum(self.motor_amplitude * motor, axis=1))), axis=1, keepdims=True)
        return np.hstack((x, y))

    def display(self, motor):
        """
        Displays the position associated with a motor configuration
        """

        # get the joints positions
        x = np.cumsum(np.multiply(self.segments_length, np.cos(np.cumsum(self.motor_amplitude * motor, axis=1))), axis=1)
        y = np.cumsum(np.multiply(self.segments_length, np.sin(np.cumsum(self.motor_amplitude * motor, axis=1))), axis=1)

        # add the agent's base
        x = np.hstack((np.zeros((x.shape[0], 1)), x))
        y = np.hstack((np.zeros((y.shape[0], 1)), y))

        # unify the cases where motor is a unique input or multiple inputs
        motor = motor.reshape((-1, self.n_motors))

        # display the different motor configurations
        for i in range(motor.shape[0]):
            plt.plot(x[i, :], y[i, :], '-o')
        plt.axis("equal")


class HingeArm6dof(HingeArm):
    """
    A four-segment arm with 4 hinge and 2 translational joints that can move its end-effector in a 2D space.
    The arm has four segments of size 9 and covers a working space of radius 36.
    Note that only the position of the end-effector is considered; its orientation is considered fixed.
    Attributes
    ----------
    motor_amplitude : list of float
        scale by which the motor components are multiplied
    segments_length : list/tuple of 3 float/int
        lengths of the arm segments
    """

    def __init__(self):
        super().__init__(type_agent="HingeArm6dof", n_motors=6, size_regular_grid=5**6)
        self.motor_amplitude = [np.pi, np.pi, np.pi, np.pi, 1, 1]
        self.segments_length = [0.375] * 4  # the arm covers a working space of radius 1.5 in an environment of size size 7

    def get_position_from_motor(self, motor):
        """
        Get the coordinates of the sensor via trigonometry.
        """
        seg_lengths = np.array(self.segments_length) * np.hstack((np.ones((motor.shape[0], 1)),
                                                                  self.motor_amplitude[4:] * motor[:, 4:],
                                                                  np.ones((motor.shape[0], 1))))
        x = np.sum(np.multiply(seg_lengths, np.cos(np.cumsum(self.motor_amplitude[0:4] * motor[:, 0:4], axis=1))), axis=1, keepdims=True)
        y = np.sum(np.multiply(seg_lengths, np.sin(np.cumsum(self.motor_amplitude[0:4] * motor[:, 0:4], axis=1))), axis=1, keepdims=True)
        return np.hstack((x, y))

    def display(self, motor):
        """
        Displays the position associated with a motor configuration
        """

        # get the joints positions
        seg_lengths = np.array(self.segments_length) * np.hstack((np.ones((motor.shape[0], 1)),
                                                                  self.motor_amplitude[4:] * motor[:, 4:],
                                                                  np.ones((motor.shape[0], 1))))
        x = np.cumsum(np.multiply(seg_lengths, np.cos(np.cumsum(self.motor_amplitude[0:4] * motor[:, 0:4], axis=1))), axis=1)
        y = np.cumsum(np.multiply(seg_lengths, np.sin(np.cumsum(self.motor_amplitude[0:4] * motor[:, 0:4], axis=1))), axis=1)

        # add the agent's base
        x = np.hstack((np.zeros((x.shape[0], 1)), x))
        y = np.hstack((np.zeros((y.shape[0], 1)), y))

        # unify the cases where motor is a unique input or multiple inputs
        motor = motor.reshape((-1, self.n_motors))

        # display the different motor configurations
        for i in range(motor.shape[0]):
            plt.plot(x[i, :], y[i, :], '-o')
