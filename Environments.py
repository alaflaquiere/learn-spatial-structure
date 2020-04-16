import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import _pickle as cpickle
from tqdm import tqdm
import gqn_renderer as gqn
import gqn_renderer.tools as tools
import gqn_renderer.bullet_tools as bullet_tools
from gqn_renderer.bullet.camera import *

"""
Collection of environments that can be used by generate-sensorimotor-data.py.
Environments are used to generate a sensory input for each position of the sensor (the environment thus implicitly includes information about the 
agent's sensor).
"""


class Environment:
    """
        type: str
            type of environment
        n_sensations: int
            dimension of the sensory input produced at each sensor position
        environment_size: [int, int] or [float, float]
            size of the environment

        get_sensation_at_position(position):
            generate the sensory inputs associated with input holistic sensor positions
        generate_shift(k):
            returns k random shifts of the environment
        display():
            displays the environment
        log(dir):
            logs the environment's parameters
    """

    def __init__(self, type_environment, n_sensations, environment_size):
        self.type = type_environment
        self.n_sensations = n_sensations
        self.environment_size = environment_size

    def get_sensation_at_position(self, position):
        return None

    def generate_shift(self, k):
        return None

    def display(self, show):
        return None

    def destroy(self):
        return None

    def save(self, destination):
        """
         Writes the environment's attributes to the disk.
         """

        try:
            serializable_dict = self.__dict__.copy()
            for key, value in self.__dict__.items():
                # keep only the ints, tuples, lists, and ndarrays
                if type(value) not in (str, int, tuple, list, np.ndarray):
                    del serializable_dict[key]
                    continue
                # make the ndarrays serializable
                if type(value) is np.ndarray:
                    serializable_dict[key] = value.tolist()
            with open(destination + "/environment_params.txt", "w") as file:
                json.dump(serializable_dict, file, indent=1)

            # save the object on disk
            with open(destination + "/environment.pkl", "wb") as f:
                cpickle.dump(self, f)

            # save an image of the environment
            fig = self.display(show=False)
            fig.savefig(destination + "/environment_image.png")
            plt.close(fig)

        except:
            print("ERROR: saving the environment in {} failed".format(destination))
            return False


class GridWorld(Environment):
    """
    Tore gridworld of size (10, 10) with a sensory input of dimension 4 at each position of the grid.
    Each sensory component is generating using a random smooth periodic function of period 10 in both x and y in the grid.
    Attributes
    ----------
    tore : bool
        make the grid behave like a tore or not
    pos2value_mapping : ndarray
        mapping from position to sensations
    """

    def __init__(self, tore=True):
        super().__init__(type_environment="GridWorld", n_sensations=4, environment_size=(10, 10))
        self.tore = tore
        self.pos2value_mapping = self.create_discrete_mapping_2dpos_to_4dsensation()

    def create_discrete_mapping_2dpos_to_4dsensation(self):
        """
        Generates a random smooth periodic function of period 10 in both x and y in the grid. Each function associated with one of
        the four sensory components is the composition of 3 random cosine varying along x and 3 random cosine varying along y.
        Return a mapping as an array of size (environment_size[0], environment_size[1] , 4).
        """

        # scan all possible positions (in [0,1]**2)
        coordinates = np.meshgrid(np.arange(0, 1, 1/self.environment_size[0]),
                                  np.arange(0, 1, 1/self.environment_size[1]))

        # create the pos2sensation_mapping
        pos2sensation_mapping = np.full((len(coordinates[0][0]), len(coordinates[0][1]), self.n_sensations), np.nan)
        for i in range(self.n_sensations):

            # draw random parameters (and ensure every even parameter is not too small)
            params = 4 * np.random.rand(12) - 2
            params[::2] = [0.25 * np.sign(val) if np.abs(val) < 0.25 else val for val in params[::2]]

            # generate the i-th sensation for all positions
            pos2sensation_mapping[:, :, i] \
                = 1 / params[0]  * np.cos(2 * np.pi * (np.round(params[0])  * coordinates[0] + params[1])) \
                + 1 / params[2]  * np.cos(2 * np.pi * (np.round(params[2])  * coordinates[0] + params[3])) \
                + 1 / params[4]  * np.cos(2 * np.pi * (np.round(params[4])  * coordinates[0] + params[5])) \
                + 1 / params[6]  * np.cos(2 * np.pi * (np.round(params[6])  * coordinates[1] + params[7])) \
                + 1 / params[8]  * np.cos(2 * np.pi * (np.round(params[8])  * coordinates[1] + params[9])) \
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

            position[:, 0] = position[:, 0] % self.environment_size[0]
            position[:, 1] = position[:, 1] % self.environment_size[1]

            sensations = self.pos2value_mapping[position[:, 0], position[:, 1]]

        else:  # returns np.nan sensations for positions outside the grid

            valid_index = (position[:, 0] >= 0) & (position[:, 0] < self.environment_size[0]) &\
                          (position[:, 1] >= 0) & (position[:, 1] < self.environment_size[1])

            sensations = np.full((position.shape[0], self.n_sensations), np.nan)
            sensations[valid_index, :] = self.pos2value_mapping[position[valid_index, 0], position[valid_index, 1]]

        if display:
            for i in tqdm(range(position.shape[0]), desc="GridWorld", mininterval=1):
                plt.cla()
                plt.imshow(sensations[[i], :], interpolation="none")
                plt.pause(1e-10)

        return sensations

    def generate_shift(self, k=1, static=False):
        """
        Returns k random shifts for the environment in [-5, 4]².
        if static=True, returns the default shift which is self.env_size/2.
        """
        if static:
            shift = (np.array(self.environment_size)//2) * np.ones((k, 2), dtype=int)
        else:
            shift = np.hstack((np.random.randint(-self.environment_size[0], self.environment_size[0], (k, 1)),
                               np.random.randint(-self.environment_size[1], self.environment_size[1], (k, 1))))
        return shift

    def display(self, show=True):
        """
        Independently display the components of the sensations in the grid.
        ax - axe where to draw the surfaces
        """
        fig = plt.figure(figsize=(12, 5))
        ax0 = fig.add_subplot(131, projection="3d")
        ax22 = [fig.add_subplot(232),
                fig.add_subplot(233),
                fig.add_subplot(235),
                fig.add_subplot(236)]
        xx, yy = np.meshgrid(np.arange(self.environment_size[0]), np.arange(self.environment_size[1]))

        for i in range(self.n_sensations):
            ax0.plot_surface(xx, yy, self.pos2value_mapping[:, :, i], alpha=0.5)

            img = ax22[i].imshow(self.pos2value_mapping[:, :, i])
            fig.colorbar(img, ax=ax22[i])

        if show:
            plt.show()

        return fig


class GQNRoom(Environment):
    """
    A 3D room of size (7,7) randomly filled with random objects. The position (0,0) corresponds to the center of the room.
    At each position, the environment generates a sensory input corresponding to the reading of a RGB camera with a fixed orientation.
    Code adapted from https://github.com/musyoku/gqn-dataset-renderer
    Attributes
    ----------
    n_obstacles : int
        number of obstacles in the environment
    """

    def __init__(self, n_obstacles=16):
        super().__init__(type_environment="3dRoom", n_sensations=16*16*3, environment_size=(7, 7))
        self.n_obstacles = n_obstacles
        self.scene = tools.build_scene(fix_light_position=True)
        # create the objects
        tools.place_objects(self.scene, tools.get_colors(),
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
        camera_height = 1.6
        camera_direction = np.array([2.5, 1.8, 0])

        # deal with the case of a single position
        position = position.reshape(-1, 2)

        # prepare variable
        sensations = np.full((position.shape[0], self.n_sensations), np.nan)

        # create the camera
        perspective_camera = gqn.pyrender.PerspectiveCamera(yfov=np.pi / 4)
        perspective_camera_node = gqn.pyrender.Node(camera=perspective_camera, translation=np.array([0, 1, 1]))

        # create the renderer
        renderer = gqn.pyrender.OffscreenRenderer(viewport_width=16, viewport_height=16)

        # add the camera in the environment
        self.scene.add_node(perspective_camera_node)

        # set the camera orientation
        yaw, pitch = tools.compute_yaw_and_pitch(camera_direction)
        perspective_camera_node.rotation = tools.generate_camera_quaternion(yaw, pitch)

        if display:
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(1, 1, 1)

        for i in tqdm(range(position.shape[0]), desc="GQNRoom", mininterval=1):

            # set the camera position
            camera_position = [position[i, 0], camera_height, position[i, 1]]
            perspective_camera_node.translation = camera_position

            # render
            image = renderer.render(self.scene, flags=gqn.pyrender.RenderFlags.SHADOWS_DIRECTIONAL)[0]

            # save sensation
            sensations[i, :] = image.reshape(-1)

            # clean the axis and display the image
            if display:
                ax.cla()
                ax.imshow(image, interpolation="none")
                plt.pause(1e-10)

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
            shift = np.array(self.environment_size)/2 * np.random.rand(k, 2) - np.array(self.environment_size)/4
        return shift

    def display(self, show=True):

        camera_position = [8, 8, 8]
        camera_direction = np.array((5, 4.7, 5))
        resolution = 512

        # create the camera
        perspective_camera = gqn.pyrender.PerspectiveCamera(yfov=np.pi / 4)
        perspective_camera_node = gqn.pyrender.Node(camera=perspective_camera, translation=np.array([0, 1, 1]))

        # create the renderer
        renderer = gqn.pyrender.OffscreenRenderer(viewport_width=resolution, viewport_height=resolution)

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(1, 1, 1)
        plt.tight_layout()

        # add the camera in the environment
        self.scene.add_node(perspective_camera_node)

        # set the camera position
        perspective_camera_node.translation = camera_position

        # set the camera orientation
        yaw, pitch = tools.compute_yaw_and_pitch(camera_direction)
        perspective_camera_node.rotation = tools.generate_camera_quaternion(yaw, pitch)

        # render
        image = renderer.render(self.scene, flags=gqn.pyrender.RenderFlags.SHADOWS_DIRECTIONAL)[0]

        # remove the camera
        self.scene.remove_node(perspective_camera_node)

        # clean the axis and display the image
        ax.cla()
        ax.imshow(image, interpolation="none")
        ax.axis("off")
        if show:
            plt.show()

        return fig


class GQNBulletRoom(Environment):
    """
    A 3D room of size (7,7) randomly filled with random objects. The position (0,0) corresponds to the center of the room.
    At each position, the environment generates a sensory input corresponding to the reading of a RGB camera with a fixed orientation.
    Code adapted from https://github.com/musyoku/gqn-dataset-renderer, and embedded in a bullet environment
    Attributes
    ----------
    n_obstacles : int
        number of obstacles in the environment
    """
    # TODO: add skybox

    def __init__(self, n_obstacles=16):
        super().__init__(
            type_environment="3dRoom",
            n_sensations=16*16*3,
            environment_size=(7, 7))
        self.n_obstacles = n_obstacles

        # Build the scene
        bullet_tools.build_scene(fix_light_position=True)

        # Create the objects
        bullet_tools.place_objects(
            bullet_tools.get_colors(),
            min_num_objects=self.n_obstacles,
            max_num_objects=self.n_obstacles,
            discrete_position=True,
            rotate_object=True)

        # Create the camera
        self.camera = Camera(45, CameraResolution(16, 16))
        self.camera.setTranslation([0, -1, 1])

    def get_sensation_at_position(self, position, display=False):
        """
        Returns the sensations at a given set of input positions.
        Inputs:
            position - (N, 2) array
        Returns:
            sensations - (self.n_sensations, 4) array
        """

        camera_height = 1.6
        camera_direction = np.array([2.5, 1.8, 0])

        # Deal with the case of a single position
        position = position.reshape(-1, 2)

        # Prepare variable
        sensations = np.full((position.shape[0], self.n_sensations), np.nan)

        # set the camera orientation
        yaw, pitch = bullet_tools.compute_yaw_and_pitch(camera_direction)
        self.camera.setOrientation(pybullet.getQuaternionFromEuler(
            [0.0, pitch, yaw]))

        if display:
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(1, 1, 1)

        for i in tqdm(range(position.shape[0]), desc="GQNRoom", mininterval=1):

            # set the camera position
            camera_position = [position[i, 0], camera_height, position[i, 1]]
            self.camera.setTranslation(bullet_tools.transform_pos_for_bullet(camera_position))

            # render
            image = self.camera.getFrame()

            # save sensation
            sensations[i, :] = image.reshape(-1)

            # clean the axis and display the image
            if display:
                ax.cla()
                ax.imshow(image, interpolation="none")
                plt.pause(1e-10)

        if display:
            plt.close(fig)
            plt.pause(0.00001)

        return sensations

    def generate_shift(self, k=1, static=False):
        """
        Returns k random shifts for the environment in [-1.75, 1.75]^2.
        If static=True, returns the default shift which is [0, 0].
        """
        if static:
            shift = np.zeros((k, 2))
        else:
            shift = np.array(self.environment_size)/2 * np.random.rand(k, 2) - np.array(self.environment_size)/4
        return shift

    def destroy(self):
        """
        Disconnect the pybullet scene.
        """
        bullet_tools.tear_down_scene()

    def display(self, show=True):
        camera_position = [8, 8, 8]
        camera_direction = np.array((5, 4.7, 5))
        resolution = 512

        overview_camera = Camera(45, CameraResolution(resolution, resolution))

        # set the camera orientation and position
        yaw, pitch = bullet_tools.compute_yaw_and_pitch(camera_direction)
        overview_camera.setOrientation(pybullet.getQuaternionFromEuler([0.0, pitch, yaw]))
        overview_camera.setTranslation(bullet_tools.transform_pos_for_bullet(camera_position))

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        plt.tight_layout()
        ax.axis("off")

        image = overview_camera.getFrame()
        # display the image
        ax.cla()
        ax.imshow(image, interpolation="none")
        ax.axis("off")
        if show:
            plt.show()
            plt.pause(0.00001)

        return fig
