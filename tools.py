import shutil
import os
import sys
import numpy as np
import _pickle as cpickle
import subprocess


def check_directory_exists(directory):
    if not os.path.exists(directory):
        print("ERROR: the directory {} doesn't exist.".format(directory))
        sys.exit()


def create_directory(directory, safe=True):
    """Create the directory or ask permission to overwrite it if it already exists"""
    if os.path.exists(directory) and safe:
        ans = input("> WARNING: The folder {} already exists; do you want to overwrite its content? [y,n]: ".format(directory))
        if ans in ["y", "Y", "yes", "Yes", "YES"]:
            shutil.rmtree(directory)
        else:
            print("exiting the program")
            sys.exit()
    os.makedirs(directory)
    return True


def get_git_hash():
    binary_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
    return binary_hash.decode("utf-8")


def load_sensorimotor_transitions(data_directory, n_transitions=None):
    """
    Loads sensorimotor transitions from a file created by generate_sensorimotor_data.py.
    Returns the data in a dictionary.
    """

    # check dir_data
    check_directory_exists(data_directory)

    print("loading sensorimotor data from {}...".format(data_directory))

    with open(data_directory, 'rb') as f:
        data = cpickle.load(f)

    # identify potential NaN entries
    to_discard = np.argwhere(np.logical_or(np.isnan(data["sensor_t"][:, 0]), np.isnan(data["sensor_tp"][:, 0])))

    # remove NaN entries
    for i in ["motor_t", "sensor_t", "shift_t", "motor_tp", "sensor_tp", "shift_tp"]:
        data[i] = np.delete(data[i], to_discard, axis=0)

    # get the number of transitions
    k = data["motor_t"].shape[0]

    # reduce the size of the dataset if necessary
    if n_transitions is None:
        n_transitions = k
    elif n_transitions < k:
        to_discard = np.arange(n_transitions, k)
        for i in ["motor_t", "sensor_t", "shift_t", "motor_tp", "sensor_tp", "shift_tp"]:
            data[i] = np.delete(data[i], to_discard, axis=0)
    else:
        n_transitions = k
        print("Warning: the requested number of data is greater than the size of the dataset.")

    print("loaded {} sensorimotor data".format(n_transitions))

    return data


def normalize_data(data):
    """
    Normalize the data such that the motor and sensor components are in [-1, 1]
    We don't normalize the positions of the sensor and shift of the environment, to keep the real scale of the external space.
    """

    # get the min/max of the motor configurations, sensations, and shifts
    motor_min = np.nanmin(data["motor_t"], axis=0)
    motor_max = np.nanmax(data["motor_t"], axis=0)
    #
    sensor_min = np.nanmin(data["sensor_t"], axis=0)
    sensor_max = np.nanmax(data["sensor_t"], axis=0)

    # normalize the data in [-1, 1]
    data["motor_t"] = 2 * (data["motor_t"] - motor_min) / (motor_max - motor_min) - 1
    data["motor_tp"] = 2 * (data["motor_tp"] - motor_min) / (motor_max - motor_min) - 1
    #
    data["sensor_t"] = 2 * (data["sensor_t"] - sensor_min) / (sensor_max - sensor_min) - 1
    data["sensor_tp"] = 2 * (data["sensor_tp"] - sensor_min) / (sensor_max - sensor_min) - 1

    # normalize the grid of motor configurations
    data["grid_motor"] = 2 * (data["grid_motor"] - motor_min) / (motor_max - motor_min) - 1

    return data

