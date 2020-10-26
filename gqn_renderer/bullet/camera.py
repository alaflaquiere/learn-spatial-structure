import pybullet
import numpy as np


class CameraResolution:
    """
    Structure for the camera resolutions
    """

    def __init__(self, width, height):
        """
        Constructor

        Parameters:
            width - Width resolution in pixels
            height - Height resolution in pixels
        """
        self.width = width
        self.height = height

    def __eq__(self, resolution):
        """
        Overloading the equal operator

        Parameters:
            resolution - the comparing resolution
        """
        try:
            assert self.width == resolution.width
            assert self.height == resolution.height
            return True

        except AssertionError:
            return False


class Camera:
    """
    Defines a camera in the environment
    """

    def __init__(self, fov, resolution):
        """
        Constructor. The aspect ratio of the camera is 1 (square image)

        Parameters:
            fov - The fov of the camera, float or np.float
            resolution - The resolution of the camera, CameraResolution object
        """
        self.setTranslation([0, 0, 1])
        self.setOrientation([0, 0, 0, 1])
        self.resolution = resolution
        self.projection_matrix = pybullet.computeProjectionMatrixFOV(
            fov,
            1,
            0.01,
            100)

        self.rgb_image = rgb_image = np.zeros((
            self.resolution.height,
            self.resolution.width,
            3))

    def setTranslation(self, translation):
        """
        Sets the translation of the camera in the world frame
        """
        self.translation = translation

    def setOrientation(self, quaternion):
        """
        Sets the rotation of the camera in the world frame
        """
        rotation = pybullet.getMatrixFromQuaternion(quaternion)
        self.forward_vector = [rotation[0], rotation[3], rotation[6]]
        self.up_vector = [rotation[2], rotation[5], rotation[8]]

        self.camera_target = [
            self.translation[0] + self.forward_vector[0] * 10,
            self.translation[1] + self.forward_vector[1] * 10,
            self.translation[2] + self.forward_vector[2] * 10]

    def setPosition(self, translation, quaternion):
        """
        Sets the translation and rotation of the camera in the world frame
        """
        self.setTranslation(translation)
        self.setOrientation(quaternion)

    def getFrame(self):
        """
        Captures an image
        """
        view_matrix = pybullet.computeViewMatrix(
            self.translation,
            self.camera_target,
            self.up_vector)

        camera_image = pybullet.getCameraImage(
            self.resolution.width,
            self.resolution.height,
            view_matrix,
            self.projection_matrix,
            shadow=1,
            lightDirection=[-0.5, -0.4, 1],
            lightDistance=20,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
            flags=pybullet.ER_NO_SEGMENTATION_MASK)

        camera_image = np.reshape(
            camera_image[2],
            (camera_image[1], camera_image[0], 4))

        self.rgb_image[:, :, 0] =\
            (1 - camera_image[:, :, 3]) * camera_image[:, :, 0] +\
            camera_image[:, :, 3] * camera_image[:, :, 0]

        self.rgb_image[:, :, 1] =\
            (1 - camera_image[:, :, 3]) * camera_image[:, :, 1] +\
            camera_image[:, :, 3] * camera_image[:, :, 1]

        self.rgb_image[:, :, 2] =\
            (1 - camera_image[:, :, 3]) * camera_image[:, :, 2] +\
            camera_image[:, :, 3] * camera_image[:, :, 2]

        frame = self.rgb_image.astype(np.uint8)
        return frame.copy()
