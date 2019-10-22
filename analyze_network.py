from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
from argparse import ArgumentParser
from tools import *


def load_subdirectory_data(dir_exp, explo_type, run_index=None):
    """
    Loads the data for the subfolders of dir_exp.
    Inputs:
        dir_exp - directory of the experiment
        explo_type - type of exploration for which to load the data
    """

    # list the subdirectories
    search_base = "run*" if run_index is None else "run" + "{:03}".format(run_index)
    sub_list = sorted(glob.glob("/".join((dir_exp, explo_type, search_base))))
    print("{} runs found for the {} type of exploration in {}".format(len(sub_list), explo_type, dir_exp))

    # initialize variables
    var = {"all_epochs": [],
           "all_losses": [],
           "all_metric_errors": [],
           "all_topo_errors_in_P": [],
           "all_topo_errors_in_H": []}

    for sub_dir in sub_list:

        # recover the Tensorboard logs
        log_file = glob.glob(sub_dir + "/tb_logs/*")[0]
        event_acc = EventAccumulator(log_file)
        event_acc.Reload()

        # extract and store the variables
        _, epochs, losses = zip(*event_acc.Scalars("loss"))
        _,      _, topo_errors_in_P = zip(*event_acc.Scalars("topology_error_in_P_1"))
        _,      _, topo_errors_in_H = zip(*event_acc.Scalars("topology_error_in_H_1"))
        _,      _, metric_errors = zip(*event_acc.Scalars("metric_error_1"))
        var["all_epochs"] += [epochs]
        var["all_losses"] += [losses]
        var["all_topo_errors_in_P"] += [topo_errors_in_P]
        var["all_topo_errors_in_H"] += [topo_errors_in_H]
        var["all_metric_errors"] += [metric_errors]

    # check that all runs are valid (they have compatible numbers of epochs)
    to_delete = []
    length_all_epochs = [len(x) for x in var["all_epochs"]]
    max_length = max(length_all_epochs)
    for ind, length in enumerate(length_all_epochs):
        if length < max_length:
            to_delete.append(ind)
            print("!! Warning: the run {} has {} epochs values instead of {} - it is discarded".
                  format(ind, length, max_length))
    # remove the entries that don't have the correct number of epochs
    for key in var.keys():
        var[key] = [val for ind, val in enumerate(var[key]) if ind not in to_delete]

    # convert the lists to arrays
    for key in var.keys():
        var[key] = np.array(var[key])

    # get the number of valid runs
    number_runs = var["all_epochs"].shape[0]
    print("{} runs loaded successfully for the {} exploration".format(number_runs, explo_type))

    return var, number_runs


def read_and_display_results(dir_exp, log_scale=False):
    """
    Plot the stats associated with an experiment. Compute the mean and std over all the runs in dir_exp.
    Inputs:
        dir_exp - directory of the experiment
        log_scale - controls if the y-axis is set to a log scale for display
    """

    # check that dir_exp exists
    check_directory_exists(dir_exp)

    # check which type of exploration exists
    exploration_types = [name for name in ["MEM", "MM", "MME"] if os.path.exists(dir_exp + "/" + name)]
    colors = {"MEM": "r", "MM": "g", "MME": "b"}

    # prepare the figure
    fig = plt.figure(dir_exp, figsize=(16, 4))
    ax1 = fig.add_subplot(141)
    ax1.set_title('loss')
    ax2 = fig.add_subplot(142)
    ax2.set_title('$D_{topo in P}$')
    ax3 = fig.add_subplot(143)
    ax3.set_title('$D_{topo}$')
    ax4 = fig.add_subplot(144)
    ax4.set_title('$D_{metric}$')

    for explo_type in exploration_types:

        # load all the data from the runs of the given type of exploration
        var, number_runs = load_subdirectory_data(dir_exp, explo_type)

        # compute stats
        losses_mean, losses_std = np.mean(var["all_losses"], axis=0), np.std(var["all_losses"], axis=0)
        topo_errors_in_P_mean, topo_errors_in_P_std = np.mean(var["all_topo_errors_in_P"], axis=0), np.std(var["all_topo_errors_in_P"], axis=0)
        topo_errors_in_H_mean, topo_errors_in_H_std = np.mean(var["all_topo_errors_in_H"], axis=0), np.std(var["all_topo_errors_in_H"], axis=0)
        metric_errors_mean, metric_errors_std = np.mean(var["all_metric_errors"], axis=0), np.std(var["all_metric_errors"], axis=0)

        # plot the variable evolution for each run
        for run in range(number_runs):
            ax1.plot(var["all_epochs"][run, :], var["all_losses"][run, :], color=colors[explo_type], alpha=0.1)
            ax2.plot(var["all_epochs"][run, :], var["all_topo_errors_in_P"][run, :], color=colors[explo_type], alpha=0.1)
            ax3.plot(var["all_epochs"][run, :], var["all_topo_errors_in_H"][run, :], color=colors[explo_type], alpha=0.1)
            ax4.plot(var["all_epochs"][run, :], var["all_metric_errors"][run, :], color=colors[explo_type], alpha=0.1)

        # plot the stats
        ax1.plot(var["all_epochs"][0, :], losses_mean, '-', color=colors[explo_type], label=explo_type)
        ax1.fill_between(var["all_epochs"][0, :], losses_mean - losses_std, losses_mean + losses_std,
                         facecolors=colors[explo_type], alpha=0.3)
        ax1.legend()
        ax1.set_yscale("log") if log_scale else None
        #
        ax2.plot(var["all_epochs"][0, :], topo_errors_in_P_mean, '-', color=colors[explo_type], label=explo_type)
        ax2.fill_between(var["all_epochs"][0, :], topo_errors_in_P_mean - topo_errors_in_P_std, topo_errors_in_P_mean + topo_errors_in_P_std,
                         facecolors=colors[explo_type], alpha=0.3)
        ax2.legend()
        ax2.set_yscale("log") if log_scale else None
        #
        ax3.plot(var["all_epochs"][0, :], topo_errors_in_H_mean, '-', color=colors[explo_type], label=explo_type)
        ax3.fill_between(var["all_epochs"][0, :], topo_errors_in_H_mean - topo_errors_in_H_std, topo_errors_in_H_mean + topo_errors_in_H_std,
                         facecolors=colors[explo_type], alpha=0.3)
        ax3.legend()
        ax3.set_yscale("log") if log_scale else None
        #
        ax4.plot(var["all_epochs"][0, :], metric_errors_mean, '-', color=colors[explo_type], label=explo_type)
        ax4.fill_between(var["all_epochs"][0, :], metric_errors_mean - metric_errors_std, metric_errors_mean + metric_errors_std,
                         facecolors=colors[explo_type], alpha=0.3)
        ax4.legend()
        ax4.set_yscale("log") if log_scale else None

    plt.show()

    return fig


def display_all_projections_of_a_single_run(dir_exp, explo_type, run_index):
    """
    Display the motor states, motor representations, and sensory states associated with a trained neural network.
    Inputs:
        dir_exp - directory of the network model
        explo_type - type of exploration for which to load the data
        run - index of the run to display
    """

    # create the path to the file
    file = "/".join([dir_exp, explo_type, "run{:03}".format(run_index), "display_progress", "display_data.pkl"])

    # check the file exists
    check_directory_exists(file)

    # load the data
    with open(file, 'rb') as f:
        data = cpickle.load(f)

    # get useful dimensions
    dim_motor, dim_sensor, dim_encoding = data["motor"].shape[1], data["gt_sensation"].shape[1], data["encoded_motor"].shape[1]

    # open the figure
    fig = plt.figure(file, figsize=(16, 4))
    # create the axis for the motor space
    ax1 = fig.add_subplot(141) if dim_motor in (1, 2) else fig.add_subplot(141, projection='3d')
    # create the axis for the encoding space
    ax2 = fig.add_subplot(142) if dim_encoding in (1, 2) else fig.add_subplot(142, projection='3d')
    # create the axis for the egocentric position
    ax3 = fig.add_subplot(143)
    # create the axis for the sensory space
    ax4 = fig.add_subplot(144) if dim_sensor in (1, 2) else fig.add_subplot(144, projection='3d')

    # plot the motor configurations
    ax1.cla()
    ax1.set_title("motor space")
    if dim_motor == 1:
        ax1.plot(data["motor"][:, 0], 0 * data["motor"][:, 0], 'b.')
        ax1.set_xlabel('$m_1$')
    elif dim_motor == 2:
        ax1.plot(data["motor"][:, 0], data["motor"][:, 1], 'b.')
        ax1.set_xlabel('$m_1$')
        ax1.set_ylabel('$m_2$')
    elif dim_motor >= 3:
        ax1.plot(data["motor"][:, 0], data["motor"][:, 1], data["motor"][:, 2], 'b.')
        ax1.set_xlabel('$m_1$')
        ax1.set_ylabel('$m_2$')
        ax1.set_zlabel('$m_3$')
    ax1.axis('equal')

    # plot the encoded motor configurations
    ax2.cla()
    ax2.set_title("encoding space")
    if dim_encoding == 1:
        ax2.plot(data["encoded_motor"][:, 0], 0 * data["encoded_motor"][:, 0], 'r.')
        ax2.set_xlabel('$h_1$')
        ax2.text(0.05, 0.05, "D_topo = {:.2e}".format(data["topo_error_in_H"]), transform=ax2.transAxes,
                 fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.2))
    elif dim_encoding == 2:
        ax2.plot(data["encoded_motor"][:, 0], data["encoded_motor"][:, 1], 'r.')
        ax2.set_xlabel('$h_1$')
        ax2.set_ylabel('$h_2$')
        ax2.text(0.05, 0.05, "D_topo = {:.2e}".format(data["topo_error_in_H"]), transform=ax2.transAxes,
                 fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.2))
    elif dim_encoding >= 3:
        ax2.plot(data["encoded_motor"][:, 0], data["encoded_motor"][:, 1], data["encoded_motor"][:, 2], 'r.')
        ax2.set_xlabel('$h_1$')
        ax2.set_ylabel('$h_2$')
        ax2.set_zlabel('$h_3$')
        ax2.text(0.05, 0.05, 0.05, "D_topo = {:.2e}".format(data["topo_error_in_H"]), transform=ax2.transAxes,
                 fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.2))
    ax2.axis('equal')

    # plot the sensor positions and the linear projection of the encoded motor configurations in the same space
    ax3.cla()
    ax3.set_title("sensor position")
    for k in range(data["gt_pos"].shape[0]):
        ax3.plot((data["gt_pos"][k, 0], data["projected_encoding"][k, 0]),
                 (data["gt_pos"][k, 1], data["projected_encoding"][k, 1]), 'r-', lw=0.4)
    ax3.plot(data["gt_pos"][:, 0], data["gt_pos"][:, 1], "o", color=[0, 0, 1], mfc="none", ms=8)
    ax3.plot(data["projected_encoding"][:, 0], data["projected_encoding"][:, 1], 'r.')
    ax3.set_xlabel('$x$')
    ax3.set_ylabel('$y$')
    ax3.text(0.05, 0.95, "D_metric = " + "{:.2e}".format(data["metric_error"]), transform=ax3.transAxes,
             fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.2))
    ax3.axis('equal')

    # plot the ground-truth and predicted sensory configurations
    ax4.cla()
    ax4.set_title("sensory space")
    if dim_sensor == 1:
        ax4.plot(data["gt_sensation"][:, 0], 0 * data["gt_sensation"][:, 0], "o", color=[0, 1, 0], ms=8, mfc="non")
        ax4.plot(data["predicted_sensation"][:, 0], 0 * data["predicted_sensation"][:, 0], 'm.')
        ax4.set_xlabel('$s_1$')
        ax4.text(0.05, 0.05, "loss={:.2e}".format(data["loss"]), transform=ax4.transAxes,
                 fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.2))
    elif dim_sensor == 2:
        ax4.plot(data["gt_sensation"][:, 0], data["gt_sensation"][:, 1], "o", color=[0, 1, 0], ms=8, mfc="none")
        ax4.plot(data["predicted_sensation"][:, 0], data["predicted_sensation"][:, 1], 'm.')
        ax4.set_xlabel('$s_1$')
        ax4.set_ylabel('$s_2$')
        ax4.text(0.05, 0.05, "loss={:.2e}".format(data["loss"]), transform=ax4.transAxes,
                 fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.2))
    elif dim_sensor >= 3:
        ax4.plot(data["gt_sensation"][:, 0], data["gt_sensation"][:, 1], data["gt_sensation"][:, 2], "o", color=[0, 0.5, 0], ms=8, mfc="none")
        ax4.plot(data["predicted_sensation"][:, 0], data["predicted_sensation"][:, 1], data["predicted_sensation"][:, 2], 'm.')
        ax4.set_xlabel('$s_1$')
        ax4.set_ylabel('$s_2$')
        ax4.set_zlabel('$s_3$')
        ax4.text(0.05, 0.05, 0.05, "loss={:.2e}".format(data["loss"]), transform=ax4.transAxes,
                 fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.2))
    ax4.axis('equal')

    # display figure
    plt.show()

    return fig


def test_encoding_module():
    #TODO
    pass


def test_sensory_prediction():
    #TODO
    pass


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-d", "--dir", dest="dir_experiment", help="path the the folder of the experiment", default="model/trained_model",
                        required=True)

    args = parser.parse_args()
    dir_experiment = args.dir_experiment

    plt.ion()

    fh = read_and_display_results(dir_experiment, log_scale=False)
    fh.savefig(dir_experiment + "/curves.png")
    fh.savefig(dir_experiment + "/curves.svg")

    index_network = 0
    for exploration_type in ["MEM", "MM", "MME"]:
        fh = display_all_projections_of_a_single_run(dir_experiment, exploration_type, index_network)
        fh.savefig(dir_experiment + "/projection_" + exploration_type + "_run" + str(index_network) + ".png")
        fh.savefig(dir_experiment + "/projection_" + exploration_type + "_run" + str(index_network) + ".svg")

    plt.show(block=True)

    input("Press any key to exit the program.")
