import os
import pickle
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use("TkAgg")  # change the backend so that the figure can stay in the background


def display(file):
    """
    Displays the data from a display_data.pkl file created by the SensorimotorPredictiveNetwork.track_progress method.
    The figure refreshes every 5s, stays in the background but stays interactive.

    Argument:
        file - path to the display_data.pkl file
    """

    # interactive mode
    plt.ion()

    while True:

        # load the data
        with open(file, 'rb') as f:
            try:
                data = pickle.load(f)
            except (FileNotFoundError, IOError) as e:
                plt.pause(5)
                continue

        # get useful dimensions
        dim_motor = data["motor"].shape[1]
        dim_sensor = data["gt_sensation"].shape[1]
        dim_encoding = data["encoded_motor"].shape[1]

        # (re)open the figure if necessary
        if not plt.fignum_exists(1):

            fig = plt.figure(num=1, figsize=(18, 5))

            plt.suptitle(file + " - epoch: " + str(data["epoch"]), fontsize=14)

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

        # plot the motor configurations
        ax1.cla()
        ax1.set_title("motor space")
        if dim_motor == 1:
            ax1.plot(data["motor"][:, 0], 0 * data["motor"][:, 0], 'bo')
        elif dim_motor == 2:
            ax1.plot(data["motor"][:, 0], data["motor"][:, 1], 'bo')
        elif dim_motor == 3:
            ax1.plot(data["motor"][:, 0], data["motor"][:, 1], data["motor"][:, 2], 'bo')
        ax1.axis('equal')

        # plot the encoded motor configurations
        ax2.cla()
        ax2.set_title("encoding space")
        if dim_encoding == 1:
            ax2.plot(data["encoded_motor"][:, 0], 0 * data["encoded_motor"][:, 0], 'rx')
        elif dim_encoding == 2:
            ax2.plot(data["encoded_motor"][:, 0], data["encoded_motor"][:, 1], 'rx')
        elif dim_encoding == 3:
            ax2.plot(data["encoded_motor"][:, 0], data["encoded_motor"][:, 1], data["encoded_motor"][:, 2], 'rx')
        ax2.axis('equal')

        # plot the sensor positions and the linear projection of the encoded motor configurations in the same space
        ax3.cla()
        ax3.set_title("sensor position")
        ax3.plot(data["gt_pos"][:, 0], data["gt_pos"][:, 1], 'ko')
        ax3.plot(data["projected_encoding"][:, 0], data["projected_encoding"][:, 1], 'rx')
        ax3.text(0.05, 0.95, "topo_error={:.2e}\nmetric error={:.2e}".format(data["topo_error"], data["metric_error"]), transform=ax3.transAxes,
                 fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.2))
        ax3.axis('equal')

        # plot the ground-truth and predicted sensory configurations
        ax4.cla()
        ax4.set_title("sensory space")
        if dim_sensor == 1:
            ax4.plot(data["gt_sensation"][:, 0], 0 * data["gt_sensation"][:, 0], 'go')
            ax4.plot(data["predicted_sensation"][:, 0], 0 * data["predicted_sensation"][:, 0], 'mx')
        elif dim_sensor == 2:
            ax4.plot(data["gt_sensation"][:, 0], data["gt_sensation"][:, 1], 'go')
            ax4.plot(data["predicted_sensation"][:, 0], data["predicted_sensation"][:, 1], 'mx')
        elif dim_sensor >= 3:
            ax4.plot(data["gt_sensation"][:, 0], data["gt_sensation"][:, 1], data["gt_sensation"][:, 2], 'go')
            ax4.plot(data["predicted_sensation"][:, 0], data["predicted_sensation"][:, 1], data["predicted_sensation"][:, 2], 'mx')
        ax4.text(0.05, 0.05, 0.05, "loss={:.2e}".format(data["loss"]), transform=ax4.transAxes,
                 fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.2))
        ax4.axis('equal')

        # save the figure
        fig.savefig(os.path.dirname(file) + '/figure.png')

        # wait
        plt.pause(5)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-f", "--filename", dest="filename", help="path to the file display_data.pkl")

    args = parser.parse_args()
    filename = args.filename

    display(filename)
