import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import glob
import numpy as np
from argparse import ArgumentParser


def read_and_display_results(dir_exp, label, color="r", fig=None):

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
        fig = plt.figure(figsize=(9, 9))
    #
    ax1 = fig.add_subplot(221)
    ax1.set_title('loss')
    #
    ax2 = fig.add_subplot(222)
    ax2.set_title('$Q_{topo} in P$')
    #
    ax3 = fig.add_subplot(223)
    ax3.set_title('$Q_{topo} in H$')
    #
    ax4 = fig.add_subplot(224)
    ax4.set_title('$Q_{metric}$')

    # plot the variable evolution for each run
    for run in range(len(sub_list)):
        ax1.plot(all_epochs[run, :], all_losses[run, :], color=color, alpha=0.2)
        ax2.plot(all_epochs[run, :], all_topo_errors_in_P[run, :], color=color, alpha=0.2)
        ax3.plot(all_epochs[run, :], all_topo_errors_in_H[run, :], color=color, alpha=0.2)
        ax4.plot(all_epochs[run, :], all_metric_errors[run, :], color=color, alpha=0.2)

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

    input("Press any key to exit the program.")
