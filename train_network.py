import os
import sys
import numpy as np
import pickle
from Networks import SensorimotorPredictiveNetwork
from argparse import ArgumentParser


def load_sensorimotor_transitions(data_directory, n_transitions=None):
    """
    Loads sensorimotor transitions from a file created by generate_sensorimotor_data.py.
    TODO
    """

    # check dir_data
    if not os.path.exists(data_directory):
        print("Error: the dataset file {} doesn't exist.".format(data_directory))
        return

    print("loading sensorimotor data {}...".format(data_directory))

    with open(data_directory, 'rb') as file:
        data = pickle.load(file)

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
    TODO
    We don't normalize the positions of the sensor, to keep the real scale.
    """

    # get the min/max of the motor configurations, sensations, and shifts
    motor_min = np.nanmin(data["motor_t"], axis=0)
    motor_max = np.nanmax(data["motor_t"], axis=0)
    #
    sensor_min = np.nanmin(data["sensor_t"], axis=0)
    sensor_max = np.nanmax(data["sensor_t"], axis=0)
    #
    shift_min = np.nanmin(data["shift_t"], axis=0)
    shift_max = np.nanmax(data["shift_t"], axis=0)

    # normalize the data in [-1, 1]
    data["motor_t"] = 2 * (data["motor_t"] - motor_min) / (motor_max - motor_min) - 1
    data["motor_tp"] = 2 * (data["motor_tp"] - motor_min) / (motor_max - motor_min) - 1
    #
    data["sensor_t"] = 2 * (data["sensor_t"] - sensor_min) / (sensor_max - sensor_min) - 1
    data["sensor_tp"] = 2 * (data["sensor_tp"] - sensor_min) / (sensor_max - sensor_min) - 1
    #
    data["shift_t"] = 2 * (data["shift_t"] - shift_min) / (shift_max - shift_min) - 1
    data["shift_tp"] = 2 * (data["shift_tp"] - shift_min) / (shift_max - shift_min) - 1

    # normalize the grid of motor configurations
    data["grid_motor"] = 2 * (data["grid_motor"] - motor_min) / (motor_max - motor_min) - 1

    return data


if __name__ == "__main__":
    """
    TODO
    """

    parser = ArgumentParser()
    parser.add_argument("-dd", "--dir_data", dest="dir_data", help="path to the data", default="dataset/gridexplorer/dataset_MMT.pkl")
    parser.add_argument("-dm", "--dir_model", dest="dir_model", help="directory where to save the model", default="model/trained_model")
    parser.add_argument("-dh", "--dim_h", dest="dim_encoding", help="dimension of the motor encoding", default=3)
    parser.add_argument("-e", "--n_epochs", dest="n_epochs", help="number of epochs", type=int, default=int(1e5))
    parser.add_argument("-n", "--n_simulations", dest="n_simulations", help="number of independent simulations", type=int, default=10)
    parser.add_argument("-gpu", "--use_gpu", dest="use_gpu", help="flag to use the gpu", type=bool, default=False)

    # get arguments
    args = parser.parse_args()
    dir_data = args.dir_data
    dir_model = args.dir_model
    dim_encoding = args.dim_encoding
    n_simulations = args.n_simulations
    n_epochs = args.n_epochs
    use_gpu = args.use_gpu

    # load the data
    transitions = load_sensorimotor_transitions(dir_data)
    dim_m = transitions["motor_t"].shape[1]
    dim_s = transitions["sensor_t"].shape[1]

    # normalize the data (including regular samplings)
    transitions = normalize_data(transitions)

    # check dir_model
    if os.path.exists(dir_model):
        ans = input("> The folder {} already exists; do you want to overwrite its content? [y,n]: ".format(dir_model))
        if ans is not "y":
            print("exiting the program")
            sys.exit()
    else:
        os.makedirs(dir_model)

    # use or not the gpu
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # iterate over the runs
    for trial in range(n_simulations):

        print("################ TRIAL {} ################".format(trial))

        # create the trial subdirectory
        dir_model_trial = "/".join([dir_model, str(trial)])
        if not os.path.exists(dir_model_trial):
            os.makedirs(dir_model_trial)

        # create the network
        network = SensorimotorPredictiveNetwork(dim_motor=dim_m, dim_sensor=dim_s, dim_enc=dim_encoding, dest_model=dir_model_trial)

        # train the network
        network.full_train(n_epochs=n_epochs, data=transitions)

    input("Press any key to exit the program.")

