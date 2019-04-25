import numpy as np
import os
import sys
import Agents
import Environments
import pickle
from argparse import ArgumentParser


def generate_sensorimotor_data(agent, environment, explo_type, k, dest_data="dataset"):
    """
    TODO
    """

    # check the inputs
    # todo

    print("generating the {} data...".format(explo_type))

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
        if explo_type is 'MTM':
            shifts_t = environment.generate_shift(k - filled)
            shifts_tp = environment.generate_shift(k - filled)
        elif explo_type is 'MM':
            shifts_t = 0 * environment.generate_shift(k - filled)  # use environment.generate_shift to get the correct data type
            shifts_tp = shifts_t
        elif explo_type is 'MMT':
            shifts_t = environment.generate_shift(k - filled)
            shifts_tp = shifts_t
        else:
            shifts_t = None
            shifts_tp = None

        # compute the holistic position of the sensor
        holi_pos_t = ego_pos_t + shifts_t
        holi_pos_tp = ego_pos_tp + shifts_tp

        # get the corresponding sensations
        sensations_t = environment.get_sensation_at_position(holi_pos_t)
        sensations_tp = environment.get_sensation_at_position(holi_pos_tp)

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

    print("data generation finished")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-n", "--n_transitions", dest="n_transitions", help="number of transitions", type=int, default=int(3e6))
    parser.add_argument("-t", "--type", dest="type_simu", help="type of simulation", default="gridexplorer")
    parser.add_argument("-d", "--dir_data", dest="dir_data", help="directory where to save the data", default="dataset/grid_explorer")

    # get arguments
    args = parser.parse_args()
    n_transitions = args.n_transitions
    type_simu = args.type_simu
    dir_data = args.dir_data

    # create the agent and environment according to the type of exploration
    if type_simu == "gridexplorer":
        my_agent = Agents.GridExplorer()
        my_environment = Environments.GridWorld()

    elif type_simu == "armroom":
        my_agent = Agents.HingeArm()
        my_environment = Environments.Room()

    else:
        print("Error: invalid type of simulation")
        sys.exit()

    # check dir_data
    if os.path.exists(dir_data):
        ans = input("> The folder {} already exists; do you want to overwrite its content? [y,n]: ".format(dir_data))
        if ans is not "y":
            print("exiting the program")
            sys.exit()
    else:
        os.makedirs(dir_data)

    # run the three types of exploration: MTM, MM, MMT
    generate_sensorimotor_data(my_agent, my_environment, "MTM", n_transitions, dir_data)
    generate_sensorimotor_data(my_agent, my_environment, "MM", n_transitions, dir_data)
    generate_sensorimotor_data(my_agent, my_environment, "MMT", n_transitions, dir_data)

    input("Press any key to exit the program.")
