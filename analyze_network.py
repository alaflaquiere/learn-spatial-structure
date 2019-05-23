import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import numpy as np
import pickle
from argparse import ArgumentParser
import train_network


def read_and_display_results(dir_exp, label, color="r", fig=None):
    """
    Plot the stats associated with an experiment. Compute the mean and std over all the runs in dir_exp.

    Inputs:
        dir_exp - directory of the experiment
        label - label to display in the plot
        color - color of the plot
        fig - figure in which to display the plots
    """

    # check that dir_exp exists
    if not os.path.exists(dir_exp):
        print("ERROR: the folder {} doesn't exist.".format(dir_exp))
        return

    # list the subdirectories
    sub_list = sorted(glob.glob(dir_exp + '/*'))
    n_runs = len(sub_list)

    # initialize variables
    all_epochs = []
    all_losses = []
    all_metric_errors = []
    all_topo_errors_in_P = []
    all_topo_errors_in_H = []

    for sub_dir in sub_list:

        # get the Tensorboard log file
        log_file = glob.glob(sub_dir + "/tb_logs/*")[0]

        # recover the Tensorboard logs
        print("load model logs from {}".format(log_file))
        event_acc = EventAccumulator(log_file)
        event_acc.Reload()

        # extract and store the variables
        _, epochs, losses = zip(*event_acc.Scalars("loss"))
        _,      _, topo_errors_in_P = zip(*event_acc.Scalars("topology_error_in_P_1"))
        _,      _, topo_errors_in_H = zip(*event_acc.Scalars("topology_error_in_H_1"))
        _,      _, metric_errors = zip(*event_acc.Scalars("metric_error_1"))
        all_epochs += [epochs]
        all_losses += [losses]
        all_topo_errors_in_P += [topo_errors_in_P]
        all_topo_errors_in_H += [topo_errors_in_H]
        all_metric_errors += [metric_errors]

    # convert the lists to arrays
    all_epochs = np.array(all_epochs)
    all_losses = np.array(all_losses)
    all_topo_errors_in_P = np.array(all_topo_errors_in_P)
    all_topo_errors_in_H = np.array(all_topo_errors_in_H)
    all_metric_errors = np.array(all_metric_errors)

    # check that all epochs are compatible
    epochs_std = np.std(all_epochs, axis=0)
    if sum(epochs_std) != 0:
        print("WARNING: the recorded epochs are not identical for all the runs; the stats are incorrect")

    # compute stats
    losses_mean = np.mean(all_losses, axis=0)
    losses_std = np.std(all_losses, axis=0)
    topo_errors_in_P_mean = np.mean(all_topo_errors_in_P, axis=0)
    topo_errors_in_P_std = np.std(all_topo_errors_in_P, axis=0)
    topo_errors_in_H_mean = np.mean(all_topo_errors_in_H, axis=0)
    topo_errors_in_H_std = np.std(all_topo_errors_in_H, axis=0)
    metric_errors_mean = np.mean(all_metric_errors, axis=0)
    metric_errors_std = np.std(all_metric_errors, axis=0)

    # open a new figure if none has been provided

    if fig is None:
        fig = plt.figure(figsize=(20, 2))
        #fig = plt.figure(figsize=(18, 4))
    #
    ax1 = fig.add_subplot(141)
    ax1.set_title('loss')
    #
    ax2 = fig.add_subplot(142)
    ax2.set_title('$D_{topo in P}$')
    #
    ax3 = fig.add_subplot(143)
    ax3.set_title('$D_{topo}$')
    #
    ax4 = fig.add_subplot(144)
    ax4.set_title('$D_{metric}$')

    # plot the variable evolution for each run
    for run in range(len(sub_list)):
        ax1.plot(all_epochs[run, :], all_losses[run, :], color=color, alpha=0.1)
        ax2.plot(all_epochs[run, :], all_topo_errors_in_P[run, :], color=color, alpha=0.1)
        ax3.plot(all_epochs[run, :], all_topo_errors_in_H[run, :], color=color, alpha=0.1)
        ax4.plot(all_epochs[run, :], all_metric_errors[run, :], color=color, alpha=0.1)

    # plot the stats
    ax1.plot(all_epochs[0, :], losses_mean, '-', color=color, label=label)
    ax1.fill_between(all_epochs[0, :], losses_mean - losses_std, losses_mean + losses_std, facecolors=color, alpha=0.5)
    ax1.legend()
    #
    ax2.plot(all_epochs[0, :], topo_errors_in_P_mean, '-', color=color, label=label)
    ax2.fill_between(all_epochs[0, :], topo_errors_in_P_mean - topo_errors_in_P_std, topo_errors_in_P_mean + topo_errors_in_P_std,
                     facecolors=color, alpha=0.5)
    ax2.legend()
    #
    ax3.plot(all_epochs[0, :], topo_errors_in_H_mean, '-', color=color, label=label)
    ax3.fill_between(all_epochs[0, :], topo_errors_in_H_mean - topo_errors_in_H_std, topo_errors_in_H_mean + topo_errors_in_H_std,
                     facecolors=color, alpha=0.5)
    ax3.legend()
    #
    ax4.plot(all_epochs[0, :], metric_errors_mean, '-', color=color, label=label)
    ax4.fill_between(all_epochs[0, :], metric_errors_mean - metric_errors_std, metric_errors_mean + metric_errors_std, facecolors=color, alpha=0.5)
    ax4.legend()

    plt.show(block=False)

    return fig


def display_all_projections_of_a_single_run(dir_exp, run):
    """
    Display the motor states, motor representations, and sensory states associated with a trained neural network.

    Inputs:
        dir_exp - directory of the network model
        run - index of the run to display
    """

    figures = {}

    for explo_type in ["MEM", "MM", "MME"]:

        # create the path to the file
        file = "/".join([dir_exp, explo_type, "run{}".format(run), "display_progress", "display_data.pkl"])

        # check the file exists
        if not os.path.exists(file):
            print("Error: the file {} does not exist".format(file))
            return

        # load the data
        with open(file, 'rb') as f:
            data = pickle.load(f)

        # get useful dimensions
        dim_motor = data["motor"].shape[1]
        dim_sensor = data["gt_sensation"].shape[1]
        dim_encoding = data["encoded_motor"].shape[1]

        # open the figure
        fig = plt.figure(figsize=(16, 4))

        # create the axis for the motor space
        if dim_motor in (1, 2):
            ax1 = plt.subplot(141)
        else:
            ax1 = plt.subplot(141, projection='3d')

        # create the axis for the encoding space
        if dim_encoding in (1, 2):
            ax2 = plt.subplot(142)
        else:
            ax2 = plt.subplot(142, projection='3d')

        # create the axis for the egocentric position
        ax3 = plt.subplot(143)

        # create the axis for the sensory space
        if dim_sensor in (1, 2):
            ax4 = plt.subplot(144)
        else:
            ax4 = plt.subplot(144, projection='3d')

        # display the updated title
        plt.suptitle(file, fontsize=14)

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

        figures[explo_type] = fig

    return figures


def display_samples(data_directory, n=24):
    """
    Display random samples from a dataset (works only for 16*16*3 images).

    Inputs:
        data_directory - directory of the dataset
        n - number of random samples to display
    """

    # check the data_directory exists
    if not os.path.exists(data_directory):
        print("Error: {} doesn't exist".format(data_directory))
        return

    # load data
    transitions = train_network.load_sensorimotor_transitions(data_directory, n_transitions=n)

    # check the data
    if transitions["sensor_t"].shape[1] != 16 * 16 * 3:
        print("Error: only images can be displayed")
        return

    # open figure
    fig = plt.figure(figsize=(15, 1))

    for i in range(n):

        # create axes
        ax = fig.add_subplot(1, n, i+1)

        # draw a sample
        index = np.random.randint(transitions["sensor_t"].shape[0])
        image = transitions["sensor_t"][index, :]

        # reshape as an image
        image = np.reshape(image, [16, 16, 3]) / 255

        # display
        ax.imshow(image)
        ax.axis("off")

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

    fh = read_and_display_results(dir_experiment + "/MEM", label="MEM", color="r")
    fh = read_and_display_results(dir_experiment + "/MM", label="MM", color="g", fig=fh)
    fh = read_and_display_results(dir_experiment + "/MME", label="MME", color="b", fig=fh)

    fh.savefig(dir_experiment + "/curves.png")
    fh.savefig(dir_experiment + "/curves.svg")

    input("Press any key to exit the program.")
