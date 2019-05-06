import os
import sys
import shutil
import numpy as np
import pickle
import glob
from Networks import SensorimotorPredictiveNetwork
from argparse import ArgumentParser


def load_sensorimotor_transitions(data_directory, n_transitions=None):
    """
    Loads sensorimotor transitions from a file created by generate_sensorimotor_data.py.
    Returns the data in a dictionary.
    """

    # check dir_data
    if not os.path.exists(data_directory):
        print("ERROR: the dataset file {} doesn't exist.".format(data_directory))
        return

    print("loading sensorimotor data {}...".format(data_directory))

    with open(data_directory, 'rb') as f:
        data = pickle.load(f)

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


if __name__ == "__main__":
    """
    # TODO
    # todo: explain the deal with --n_simulations  
    """

    parser = ArgumentParser()
    parser.add_argument("-dd", "--dir_data", dest="dir_data", help="path to the data", required=True)
    parser.add_argument("-dm", "--dir_model", dest="dir_model", help="directory where to save the models", required=True)
    parser.add_argument("-dh", "--dim_h", dest="dim_encoding", help="dimension of the motor encoding", default=3)
    parser.add_argument("-e", "--n_epochs", dest="n_epochs", help="number of epochs", type=int, default=int(1e5))
    parser.add_argument("-n", "--n_simulations", dest="n_simulations",
                        help="number of independent simulations (used only if a single dataset is provided)", type=int, default=1)
    parser.add_argument("-v", "--visual", dest="display_progress", help="flag to turn the online display on or off", action="store_true")
    parser.add_argument("-gpu", "--use_gpu", dest="use_gpu", help="flag to use the gpu", action="store_true")
    parser.add_argument("-mem", "--mem", dest="mem", help="flag to run simulations on the MEM data", action="store_true")
    parser.add_argument("-mm", "--mm", dest="mm", help="flag to run simulations on the MM data", action="store_true")
    parser.add_argument("-mme", "--mme", dest="mme", help="flag to run simulations on the MME data", action="store_true")

    # get arguments
    args = parser.parse_args()
    dir_data = args.dir_data
    dir_model = args.dir_model
    dim_encoding = args.dim_encoding
    n_simulations = args.n_simulations
    n_epochs = args.n_epochs
    display_progress = args.display_progress
    use_gpu = args.use_gpu
    mem = args.mem
    mm = args.mm
    mme = args.mme

    # check dir_data
    if not os.path.exists(dir_data):
        print("ERROR: the dataset file {} doesn't exist.".format(dir_data))
        sys.exit()

    # check dir_model
    if os.path.exists(dir_model):
        ans = input("> WARNING: The folder {} already exists; do you want to overwrite its content? [y,n]: ".format(dir_model))
        if ans is "y":
            shutil.rmtree(dir_model)
        else:
            print("exiting the program")
            sys.exit()
    else:
        os.makedirs(dir_model)

    # use the gpu or not
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # check on which data to run
    simu_types = []
    if mem:
        simu_types += ["MEM"]
    if mm:
        simu_types += ["MM"]
    if mme:
        simu_types += ["MME"]

    # get the different sub-datasets
    subfolder_list = sorted(glob.glob(dir_data + "/*"))
    print("{} datasets have been found in {}".format(len(subfolder_list), dir_data))

    # check that n_simulations is compatible with subfolder_list
    if len(subfolder_list) == 1:
        print("{} runs will be performed on the dataset".format(n_simulations))
    else:
        print("1 run will be performed on each dataset")
        if n_simulations != 1:
            ans = input("> WARNING: n_simulations ({}) is overwritten to be {}; continue? [y,n]: ".format(n_simulations, len(subfolder_list)))
            if ans is "y":
                n_simulations = len(subfolder_list)
            else:
                sys.exit()

    # run the training on the different types of data
    for simu_type in simu_types:

        # iterate over the runs
        for trial in range(n_simulations):

            print("################ {} DATA - TRIAL {} ################".format(simu_type, trial))

            # get the correct file name
            filename = "{}/dataset{}/dataset_{}.pkl".format(dir_data, trial % n_simulations, simu_type)

            # load the data
            transitions = load_sensorimotor_transitions(filename)
            dim_m = transitions["motor_t"].shape[1]
            dim_s = transitions["sensor_t"].shape[1]

            # normalize the data (including regular samplings)
            transitions = normalize_data(transitions)

            # create the trial subdirectory
            dir_model_trial = "/".join([dir_model, simu_type, "run" + str(trial)])
            if not os.path.exists(dir_model_trial):
                os.makedirs(dir_model_trial)

            # create the network
            network = SensorimotorPredictiveNetwork(dim_motor=dim_m, dim_sensor=dim_s, dim_enc=dim_encoding, dest_model=dir_model_trial)

            # save the network parameters
            network.log(dir_model_trial + "/network_params.txt")

            # get and save the uuid of the training dataset
            with open(dir_data + "/uuid.txt", "r") as file:
                dataset_uuid = file.read()
            with open(dir_model_trial + "/training_dataset_uuid.txt", "w") as file:
                file.write("{} - {}".format(dir_data, dataset_uuid))

            # train the network
            network.full_train(n_epochs=n_epochs, data=transitions, disp=display_progress)

    input("Press any key to exit the program.")
