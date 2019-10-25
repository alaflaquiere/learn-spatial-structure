import os
import _pickle as cpickle
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use("TkAgg")  # backend so that the figure can stay in the background


def run_display_server(file, refresh):
    """
    Displays the data from a display_data.pkl file created by the SensorimotorPredictiveNetwork.track_progress() method.
    The figure refreshes every 5s, stays in the background but stays interactive.
    Argument:
        file - path to the display_data.pkl file
    """

    # interactive mode
    plt.ion()

    # wait for the file to be created
    while True:
        if os.path.exists(file):
            break
        plt.pause(refresh)

    while True:

        # load the data
        try:
            with open(file, 'rb') as f:
                try:
                    data = cpickle.load(f)
                except (IOError, EOFError):
                    plt.pause(refresh)
                    continue
        except FileNotFoundError:
            plt.pause(refresh)
            continue

        fig = display_data(data)

        # save the figure
        fig.savefig(os.path.dirname(file) + '/figure.png')
        fig.savefig(os.path.dirname(file) + '/figure.svg')

        # wait
        plt.pause(refresh)


def display_data(data, fig_number=1, name=""):
    """
    Displays the data from a display_data.pkl file created by the SensorimotorPredictiveNetwork.track_progress() method.
    Argument:
        data - data to display
        fig_number - index of the figure to plat in
        name - text to add to the figure
    """

    # get useful dimensions
    dim_motor = data["motor"].shape[1]
    dim_sensor = data["gt_sensation"].shape[1]
    dim_encoding = data["encoded_motor"].shape[1]

    # open the figure
    if not plt.fignum_exists(fig_number):
        fig = plt.figure(num=fig_number, figsize=(16, 5))
        # create the axis for the motor space
        ax1 = plt.subplot(141) if dim_motor in (1, 2) else plt.subplot(141, projection='3d')
        # create the axis for the encoding space
        ax2 = plt.subplot(142) if dim_motor in (1, 2) else plt.subplot(142, projection='3d')
        # create the axis for the egocentric position
        ax3 = plt.subplot(143)
        # create the axis for the sensory space
        ax4 = plt.subplot(144) if dim_motor in (1, 2) else plt.subplot(144, projection='3d')
    else:
        fig = plt.figure(num=fig_number)
        ax1, ax2, ax3, ax4 = fig.axes


    # display the updated title
    plt.suptitle(name + " - epoch: " + str(data["epoch"]), fontsize=14)

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
        ax2.text(0.05, 0.05, "topo_error_in_H={:.2e}".format(data["topo_error_in_H"]), transform=ax2.transAxes,
                 fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.2))
    elif dim_encoding == 2:
        ax2.plot(data["encoded_motor"][:, 0], data["encoded_motor"][:, 1], 'r.')
        ax2.set_xlabel('$h_1$')
        ax2.set_ylabel('$h_2$')
        ax2.text(0.05, 0.05, "topo_error_in_H={:.2e}".format(data["topo_error_in_H"]), transform=ax2.transAxes,
                 fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.2))
    elif dim_encoding >= 3:
        ax2.plot(data["encoded_motor"][:, 0], data["encoded_motor"][:, 1], data["encoded_motor"][:, 2], 'r.')
        ax2.set_xlabel('$h_1$')
        ax2.set_ylabel('$h_2$')
        ax2.set_zlabel('$h_3$')
        ax2.text(0.05, 0.05, 0.05, "topo_error_in_H={:.2e}".format(data["topo_error_in_H"]), transform=ax2.transAxes,
                 fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.2))
    ax2.axis('equal')

    # plot the sensor positions and the linear projection of the encoded motor configurations in the same space
    ax3.cla()
    ax3.set_title("sensor position")
    #
    if data["gt_pos"].shape[0] < 1000:
        for k in range(data["gt_pos"].shape[0]):
            ax3.plot((data["gt_pos"][k, 0], data["projected_encoding"][k, 0]),
                     (data["gt_pos"][k, 1], data["projected_encoding"][k, 1]), 'r-', lw=0.4)
    #
    ax3.plot(data["gt_pos"][:, 0], data["gt_pos"][:, 1], 'bo', mfc="none", ms=8)
    ax3.plot(data["projected_encoding"][:, 0], data["projected_encoding"][:, 1], 'r.')
    ax3.set_xlabel('$x$')
    ax3.set_ylabel('$y$')
    ax3.text(0.05, 0.95, "topo_error_in_P={:.2e}\nmetric error={:.2e}".format(data["topo_error_in_P"], data["metric_error"]), transform=ax3.transAxes,
             fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.2))
    ax3.axis('equal')

    # plot the ground-truth and predicted sensory configurations
    ax4.cla()
    ax4.set_title("sensory space")
    if dim_sensor == 1:
        ax4.plot(data["gt_sensation"][:, 0], 0 * data["gt_sensation"][:, 0], 'o', color=[0, 0.5, 0], ms=8, mfc="none")
        ax4.plot(data["predicted_sensation"][:, 0], 0 * data["predicted_sensation"][:, 0], 'm.')
        ax4.set_xlabel('$s_1$')
        ax4.text(0.05, 0.05, "loss={:.2e}".format(data["loss"]), transform=ax4.transAxes,
                 fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.2))
    elif dim_sensor == 2:
        ax4.plot(data["gt_sensation"][:, 0], data["gt_sensation"][:, 1], 'o', color=[0, 0.5, 0], ms=8, mfc="none")
        ax4.plot(data["predicted_sensation"][:, 0], data["predicted_sensation"][:, 1], 'm.')
        ax4.set_xlabel('$s_1$')
        ax4.set_ylabel('$s_2$')
        ax4.text(0.05, 0.05, "loss={:.2e}".format(data["loss"]), transform=ax4.transAxes,
                 fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.2))
    elif dim_sensor >= 3:
        ax4.plot(data["gt_sensation"][:, 0], data["gt_sensation"][:, 1], data["gt_sensation"][:, 2], 'o', color=[0, 0.5, 0], ms=8, mfc="none")
        ax4.plot(data["predicted_sensation"][:, 0], data["predicted_sensation"][:, 1], data["predicted_sensation"][:, 2], 'm.')
        ax4.set_xlabel('$s_1$')
        ax4.set_ylabel('$s_2$')
        ax4.set_zlabel('$s_3$')
        ax4.text(0.05, 0.05, 0.05, "loss={:.2e}".format(data["loss"]), transform=ax4.transAxes,
                 fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.2))
    ax4.axis('equal')

    return fig


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-f", "--filename", dest="filename", help="path to the file display_data.pkl")

    args = parser.parse_args()
    filename = args.filename

    run_display_server(filename, refresh=5)
