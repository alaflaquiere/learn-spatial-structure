import numpy as np
import os
import sys
import shutil
import Agents
import Environments
import pickle
from argparse import ArgumentParser
import uuid


# TODO: save an image of the environment with the data
# todo: check what is missing in the local/venv pyrender

def generate_sensorimotor_data(agent, environment, explo_type, k, dest_data="dataset", scale_static_case=False, disp=True):
    """
    Generates a sensorimotor dataset and save it in <dest_data>/dataset_<explo_type>.pkl.
    k sensorimotor transitions are generated by drawing random motor configurations and environment shifts for each sensorimotor experience.

    Inputs:
        agent - the agent generating the motor configurations and egocentric sensor positions
        environment - the environment generating the environment shifts and the sensations associated with the holistic sensor positions
        explo_type - type of exploration which changes how the shifts are generated
                     MM: the shift is always 0
                     MEM: a different shift is drawn for the first and second sensorimotor couple of each transition
                     MME: the same random shift is used for the first and second sensorimotor couple of each transition
        k - number of transitions to generate
        dest_data - directory where to save the data

    Output:
        The generated dataset is saved in <dest_data>/dataset_<explo_type>.pkl as a dictionary with the following structure:
        transitions = {"motor_t": np.array(n_transitions, agent.n_motors),
                       "sensor_t": np.array(n_transitions, environment.n_sensations),
                       "shift_t": np.array(n_transitions, 2)2,
                       "motor_tp": np.array(n_transitions, agent.n_motors),
                       "sensor_tp": np.array(n_transitions, environment.n_sensations),
                       "shift_tp": np.array(n_transitions, 2),
                       "grid_motor": np.array(agent.size_regular_grid, agent.n_motors),
                       "grid_pos": np.array(agent.size_regular_grid, 2)
                       }
    """

    print("generating {} data...".format(explo_type))

    # prepare the dictionary
    transitions = {"motor_t": np.full((k, agent.n_motors), np.nan),
                   "sensor_t": np.full((k, environment.n_sensations), np.nan),
                   "shift_t": np.full((k, 2), np.nan),
                   "motor_tp": np.full((k, agent.n_motors), np.nan),
                   "sensor_tp": np.full((k, environment.n_sensations), np.nan),
                   "shift_tp": np.full((k, 2), np.nan),
                   "grid_motor": np.full((agent.size_regular_grid, agent.n_motors), np.nan),
                   "grid_pos": np.full((agent.size_regular_grid, 2), np.nan)}

    # generate random transitions
    filled = 0
    while filled < k:

        # generate (k - filled) motor states and sensor positions
        motor_t, ego_pos_t = agent.generate_random_sampling(k - filled)
        motor_tp, ego_pos_tp = agent.generate_random_sampling(k - filled)

        # generate (k - filled) shifts of the environment
        if explo_type is 'MEM':
            shifts_t = environment.generate_shift(k - filled)
            shifts_tp = environment.generate_shift(k - filled)
        elif explo_type is 'MM':
            shifts_t = environment.generate_shift(k - filled, static=True)  # use environment.generate_shift to get the correct data type
            shifts_tp = shifts_t
        elif explo_type is 'MME':
            shifts_t = environment.generate_shift(k - filled)
            shifts_tp = shifts_t
        else:
            shifts_t = None
            shifts_tp = None

        # scale the range of exploration by a factor 2 in the MM case
        # to get comparable sensory distributions over the exploration in the three exploration cases
        if explo_type is 'MM' and scale_static_case:
            ego_pos_t = ego_pos_t * 2
            ego_pos_tp = ego_pos_tp * 2

        # compute the holistic position of the sensor
        holi_pos_t = ego_pos_t + shifts_t
        holi_pos_tp = ego_pos_tp + shifts_tp

        # get the corresponding sensations
        sensations_t = environment.get_sensation_at_position(holi_pos_t, display=disp)
        sensations_tp = environment.get_sensation_at_position(holi_pos_tp, display=disp)

        # get the indexes of valid sensory inputs
        valid_indexes = np.argwhere((~np.isnan(sensations_t[:, 0])) & (~np.isnan(sensations_tp[:, 0])))[:, 0]

        # fill the dictionary
        transitions["motor_t"][filled:filled + len(valid_indexes), :] = motor_t[valid_indexes, :]
        transitions["sensor_t"][filled:filled + len(valid_indexes), :] = sensations_t[valid_indexes, :]
        transitions["shift_t"][filled:filled + len(valid_indexes), :] = shifts_t[valid_indexes, :]
        transitions["motor_tp"][filled:filled + len(valid_indexes), :] = motor_tp[valid_indexes, :]
        transitions["sensor_tp"][filled:filled + len(valid_indexes), :] = sensations_tp[valid_indexes, :]
        transitions["shift_tp"][filled:filled + len(valid_indexes), :] = shifts_tp[valid_indexes, :]

        # update filled
        filled = filled + len(valid_indexes)

    # get a regular grid of motor and position samples
    transitions["grid_motor"], transitions["grid_pos"] = agent.generate_regular_sampling()

    # save the dictionary on disk
    filename = "dataset_{}.pkl".format(explo_type)
    with open("/".join([dest_data, filename]), 'wb') as file:
        pickle.dump(transitions, file)

    # TODO: check that the file has been correctly saved by trying to load it

    print("data generation finished")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-n", "--n_transitions", dest="n_transitions", help="number of transitions", type=int, default=150000)  # int(3e6)
    parser.add_argument("-t", "--type", dest="type_simu", help="type of simulation", choices=["gridexplorer", "armflatroom", "arm3droom"],
                        required=True)
    parser.add_argument("-r", "--n_runs", dest="n_runs", help="number of independent datasets generated", type=int, default=1)
    parser.add_argument("-d", "--dir_data", dest="dir_data", help="directory where to save the data", required=True)
    parser.add_argument("-s", "--scale_mm", dest="scale_mm",
                        help="flag to scale up the arm for the MM exploration and ensure comparable sensory distributions", action="store_true")
    parser.add_argument("-v", "--visual", dest="display_exploration", help="flag to turn the online display on or off", action="store_true")

    # get arguments
    args = parser.parse_args()
    n_transitions = args.n_transitions
    type_simu = args.type_simu
    n_runs = args.n_runs
    dir_data = args.dir_data
    scale_mm = args.scale_mm
    display_exploration = args.display_exploration

    # check dir_data
    if os.path.exists(dir_data):
        ans = input("> WARNING: The folder {} already exists; do you want to overwrite its content? [y,n]: ".format(dir_data))
        if ans is "y":
            shutil.rmtree(dir_data)
        if ans is not "y":
            print("exiting the program")
            sys.exit()
    else:
        os.makedirs(dir_data)

    # iterate over the runs
    for trial in range(n_runs):

        print("################ ENVIRONMENT {} ################".format(trial))

        # create the trial subdirectory
        dir_data_trial = "/".join([dir_data, "dataset"+str(trial)])
        if not os.path.exists(dir_data_trial):
            os.makedirs(dir_data_trial)

        # create the agent and environment according to the type of exploration
        if type_simu == "gridexplorer":
            my_agent = Agents.GridExplorer()
            my_environment = Environments.GridWorld()

        elif type_simu == "armflatroom":
            my_agent = Agents.HingeArm(segments_length=[12, 12, 12])  # working space of radius 36 in an environment of size size 150

            # if a single run is asked, the user has the freedom to select the environment
            if n_runs == 1:
                validated = False
                while not validated:
                    my_environment = Environments.FlatRoom()
                    my_environment.display()
                    ans = input("> redraw a different environment? [y, n]: ")
                    if ans is "n":
                        validated = True
            else:
                my_environment = Environments.FlatRoom()

        elif type_simu == "arm3droom":
            my_agent = Agents.HingeArm(segments_length=[0.5, 0.5, 0.5])  # working space of radius 1.5 in an environment of size size 7
            my_environment = Environments.GQNRoom()

        else:
            print("Error: invalid type of simulation")
            sys.exit()

        # save the agent and environment parameters in a readable form
        my_agent.log(dir_data_trial + "/agent_params.txt")
        my_environment.log(dir_data_trial + "/environment_params.txt")

        # save my_agent and my_environment to disk
        with open(dir_data_trial + "/agent.pkl", "wb") as file:
            pickle.dump(my_agent, file)

        if type_simu != "armflatroom":
        # todo: NEEDS TO BE REWORKED IN THE FLATLAND CASE SO THAT THE NECESSARY PARAMETERS OF THE ENVIRONMENT ARE SAVED (ABLE TO RECONSTRUCT)
            with open(dir_data_trial + "/environment.pkl", "wb") as file:
                pickle.dump(my_environment, file)

        # create and save a unique identifier for the dataset
        with open(dir_data_trial + "/uuid.txt", "w") as file:
            file.write(uuid.uuid4().hex)

        # run the three types of exploration: MEM, MM, MME
        generate_sensorimotor_data(my_agent, my_environment, "MEM",
                                   n_transitions, dir_data_trial, scale_static_case=scale_mm, disp=display_exploration)
        generate_sensorimotor_data(my_agent, my_environment, "MM",
                                   n_transitions, dir_data_trial, scale_static_case=scale_mm, disp=display_exploration)
        generate_sensorimotor_data(my_agent, my_environment, "MME",
                                   n_transitions, dir_data_trial, scale_static_case=scale_mm, disp=display_exploration)

        # TODO: save in a temporary folder while the dataset is being generated, and rename it to the correct name only once it has been entirely generated and checked

    input("Press any key to exit the program.")
