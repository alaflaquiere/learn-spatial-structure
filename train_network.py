import json
import glob
import datetime
from Networks import SensorimotorPredictiveNetwork
from argparse import ArgumentParser
import uuid
from tools import *


def save_training(directory, path_data, run, type_simu, args):
    """save a UUID for the simulation"""

    with open(path_data + "/generation_params.txt", "r") as f:
        data_params = json.load(f)

    dictionary = {"UUID": uuid.uuid4().hex,
                  "UUID source data": data_params["UUID"],
                  "Source data": path_data,
                  "Sigma noise motor": args.sigma_noise_motor,
                  "Sigma noise sensor": args.sigma_noise_sensor,
                  "Time": datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                  "Type simulation": type_simu,
                  "Run": run,
                  "Destination": directory,
                  "Nbr epochs": args.n_epochs,
                  "code commit hash": get_git_hash()}
    try:
        with open(directory + "/training_params.txt", "w") as f:
            json.dump(dictionary, f, indent=2)
    except:
        print("ERROR: saving training_params.txt in {} failed".format(directory))
        return False
    return True


if __name__ == "__main__":
    """
    Train a sensorimotor predictive network on a dataset generated by generate_sensorimotor_data.py.
    If the target dataset contains a single sub-dataset, --n_simulations networks are trained on this dataset.
    Otherwise, a single network is trained on each sub-dataset.
    -mem, -mm, and -mme can be specified to train on specific types of exploration; if none is specified, all types are considered
    """

    parser = ArgumentParser()
    parser.add_argument("-dd", "--dir_data", dest="dir_data", help="path to the data", required=True)
    parser.add_argument("-dm", "--dir_model", dest="dir_model", help="directory where to save the models", required=True)
    parser.add_argument("-dh", "--dim_h", dest="dim_encoding", help="dimension of the motor encoding", default=3)
    parser.add_argument("-e", "--n_epochs", dest="n_epochs", help="number of epochs", type=int, default=int(1e5))
    parser.add_argument("-n", "--n_simulations", dest="n_simulations",
                        help="number of independent simulations (used only if a single dataset is provided)", type=int, default=1)
    parser.add_argument("-sm", "--sigma_noise_motor", dest="sigma_noise_motor",
                        help="amplitude of the noise on the motor after normalization in [-1, 1]", type=float, default=0)
    parser.add_argument("-ss", "--sigma_noise_sensor", dest="sigma_noise_sensor",
                        help="amplitude of the noise on the sensor after normalization in [-1, 1]", type=float, default=0)
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
    sigma_noise_motor = args.sigma_noise_motor
    sigma_noise_sensor = args.sigma_noise_sensor
    n_simulations = args.n_simulations
    n_epochs = args.n_epochs
    display_progress = args.display_progress
    use_gpu = args.use_gpu
    if not(args.mem or args.mm or args.mme):  # if no exploration has been specified, do all of them
        mem = mm = mme = True
    else:
        mem, mm, mme = args.mem, args.mm, args.mme

    # check dir_data
    if not os.path.exists(dir_data):
        print("ERROR: the dataset file {} doesn't exist.".format(dir_data))
        sys.exit()

    # check the folder for the results
    create_directory(dir_model)

    # use the gpu or not
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if use_gpu else "-1"

    # check on which data to run
    simu_types = [["MEM", "MM", "MME"][i] for i, val in enumerate([mem, mm, mme]) if val is True]

    # get the different sub-datasets
    subfolder_list = sorted(glob.glob(dir_data + "/dataset*"))
    print("{} datasets have been found in {}".format(len(subfolder_list), dir_data))

    # set n_simulations to len(subfolder_list) if len(subfolder_list) is greater than 1
    if len(subfolder_list) == 1:
        print("{} runs will be performed on the dataset".format(n_simulations))
    else:
        print("1 run will be performed on each dataset")
        n_simulations = len(subfolder_list)

    # iterate over the runs
    for trial in range(n_simulations):

        print("[[TRIAL {}]]".format(trial))

        # run the training on the different types of data
        for simu_type in simu_types:

            # get the correct data folder and file name
            sub_dir_data = "{}/dataset{:03}".format(dir_data, trial % len(subfolder_list))
            filename = "{}/dataset_{}.pkl".format(sub_dir_data, simu_type)

            print("[{} EXPLORATION - (dataset: {})]".format(simu_type, filename))

            # load the data
            transitions = load_sensorimotor_transitions(filename)
            dim_m = transitions["motor_t"].shape[1]
            dim_s = transitions["sensor_t"].shape[1]

            # normalize the data (including regular samplings)
            transitions = normalize_data(transitions)

            # add noise
            for key in ["motor_t", "motor_tp"]:
                transitions[key] += sigma_noise_motor * np.random.randn(*transitions[key].shape)
            for key in ["sensor_t", "sensor_tp"]:
                transitions[key] += sigma_noise_sensor * np.random.randn(*transitions[key].shape)

            # create the trial subdirectory
            dir_model_trial = "/".join([dir_model, simu_type, "run" + "{:03}".format(trial)])
            create_directory(dir_model_trial, safe=False)

            # create the network
            network = SensorimotorPredictiveNetwork(dim_motor=dim_m, dim_sensor=dim_s, dim_enc=dim_encoding, model_destination=dir_model_trial)

            # save the network parameters
            network.save(dir_model_trial)

            # copy generation_params from the source dataset
            save_training(dir_model_trial, sub_dir_data, trial, simu_type, args)

            # train the network
            network.full_train(n_epochs=n_epochs, data=transitions, disp=display_progress)

    input("Press any key to exit the program.")
