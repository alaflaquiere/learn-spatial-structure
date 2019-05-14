import os
import time
import tensorflow as tf
import numpy as np
import pickle
from sklearn import linear_model
from scipy.spatial.distance import pdist
import platform
import subprocess
import json


class MLP:
    """
    A simple fully connected MLP class with no activation function on the last layer.
    (reuse is set to tf.AUTO_REUSE in order to allow siamese MLPs.)

    Parameters:
        input_net - node used as input to the MLP
        layers_sizes - list of the sizes of the successives layers of the MLP
        activation - activation function used in the MLP

    Attribute:
        output - output node of the MLP
    """

    def __init__(self, input_net, layers_sizes, activation=tf.nn.selu):

        self.type = "MLP"

        current_input = input_net
        for layer_index, nbr_units in enumerate(layers_sizes[0:-1]):
            current_input = tf.layers.dense(inputs=current_input,
                                            units=nbr_units,
                                            activation=activation,
                                            name="layer"+str(layer_index))
        self.output = tf.layers.dense(inputs=current_input, units=layers_sizes[-1], activation=None, name='layerend')


class SensorimotorPredictiveNetwork:
    """
    Network to perform sensory prediction based on a current motor configuration, a current sensory input, and a future motor configuration.
    Both the motor configurations are encoded using a single (siamese) module.

    Parameters:
        dim_motor - dimension of the motor configuration
        dim_sensor - dimension of the sensory input
        dim_enc - dimension of the output of the motor encoding module
        dest_model - directory in which to save the model and temporary files
    """

    def __init__(self, dim_motor=3, dim_sensor=4, dim_enc=3, dest_model="model/trained_model", act_fn="selu"):

        # set attributes
        self.type = "SensorimotorPredictiveNetwork"
        self.dim_enc = dim_enc
        self.encoding_layers_size = [150, 100, 50]
        self.predictive_layers_size = [200, 150, 100]
        self.activation = act_fn
        self.learning_rate_param = [1e-3, 1e-5, 8e4, 1]
        self.batch_size = 100
        self.dim_motor = dim_motor
        self.dim_sensor = dim_sensor
        self.dest_model = dest_model
        self.lin_reg_model = linear_model.LinearRegression(fit_intercept=True)

        # get the activation function (a temporary string is used to simply log the class attributes in self.log())
        if self.activation == "selu":
            activation = tf.nn.selu
        elif self.activation == "relu":
            activation = tf.nn.relu
        else:
            print("WARNING: Incorrect activation function - tf.nn.selu is used instead")
            activation = tf.nn.selu

        # reset the default graph
        tf.reset_default_graph()

        # create input and output placeholders
        self.motor_t = tf.placeholder(dtype=tf.float32, shape=[None, self.dim_motor], name='motor_t')
        self.motor_tp = tf.placeholder(dtype=tf.float32, shape=[None, self.dim_motor], name='motor_tp')
        self.sensor_t = tf.placeholder(dtype=tf.float32, shape=[None, self.dim_sensor], name='sensor_t')
        self.sensor_tp = tf.placeholder(dtype=tf.float32, shape=[None, self.dim_sensor], name='sensor_tp')

        # create placeholders for the dissimilarity measures
        self.metric_error = tf.placeholder(dtype=tf.float32, shape=[], name='metric_error')
        self.topology_error_in_P = tf.placeholder(dtype=tf.float32, shape=[], name='topology_error_in_P')
        self.topology_error_in_H = tf.placeholder(dtype=tf.float32, shape=[], name='topology_error_in_H')

        # define the network
        # define the motor encoding modules
        with tf.variable_scope("motor_encoding", reuse=tf.AUTO_REUSE):

            # create the first copy of the encoding module
            self.encode_module_t = MLP(input_net=self.motor_t,
                                       layers_sizes=self.encoding_layers_size + [self.dim_enc],
                                       activation=activation)

            # create the second copy of the encoding module
            self.encode_module_tp = MLP(input_net=self.motor_tp,
                                        layers_sizes=self.encoding_layers_size + [self.dim_enc],
                                        activation=activation)

        # concatenate the motor encodings with the sensory input
        concatenation = tf.concat([self.encode_module_t.output, self.encode_module_tp.output, self.sensor_t], axis=1, name='concat')

        # define the predictive module
        with tf.variable_scope("sensory_prediction", reuse=tf.AUTO_REUSE):
            self.prediction_module = MLP(input_net=concatenation, layers_sizes=self.predictive_layers_size + [self.dim_sensor])

        # define the loss
        loss = tf.reduce_sum(tf.squared_difference(self.prediction_module.output, self.sensor_tp), axis=1)
        self.loss = tf.reduce_mean(loss, axis=0)

        # define the learning rate
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.polynomial_decay(self.learning_rate_param[0], self.global_step, self.learning_rate_param[2],
                                                       self.learning_rate_param[1], power=self.learning_rate_param[3])

        # define the optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.minimize_op = optimizer.minimize(self.loss, global_step=self.global_step)

        # create trackers for Tensorboard
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("metric_error", self.metric_error)
        tf.summary.scalar("topology_error_in_P", self.topology_error_in_P)
        tf.summary.scalar("topology_error_in_H", self.topology_error_in_H)
        tf.summary.scalar("learning_rate", self.learning_rate)
        self.merged_summaries = tf.summary.merge_all()
        self.graph = tf.get_default_graph()  # get the default graph
        self.summaries_writer = tf.summary.FileWriter(self.dest_model + '/tb_logs', self.graph)

        # create a saver
        self.saver = tf.train.Saver()

        # token session
        self.sess = None

    def train(self, data, number_epochs=1):
        """
        Perform <number_epochs> iterations of training on minibatches from the transitions <data>.
        """

        # run the optimization for number_epochs iterations
        current_loss = None
        for k in range(number_epochs):

            # draw indexes with repeat (without repeat significantly increases computation time)
            batch_indexes = np.random.choice(data["motor_t"].shape[0], self.batch_size, replace=True)

            # run the optimization
            current_loss, _ = self.sess.run([self.loss, self.minimize_op], feed_dict={self.motor_t: data["motor_t"][batch_indexes, :],
                                                                                      self.sensor_t: data["sensor_t"][batch_indexes, :],
                                                                                      self.motor_tp: data["motor_tp"][batch_indexes, :],
                                                                                      self.sensor_tp: data["sensor_tp"][batch_indexes, :]})
        current_epoch = self.sess.run(self.global_step)

        return current_epoch, current_loss

    def full_train(self, n_epochs, data, disp):
        """
        Performs successive training cycles of 1000 epochs on data up to n_epochs epochs. After each cycle, evaluates the network, save the variables
        tracked in Tensorboard, and save the network.
        If disp=True, also launch display_progress.py in a parallel process to visualize the network progress.
        """

        print('training the network...')

        # open the display process in parallel if necessary
        if disp:
            if platform.system() == 'Windows':
                command = "python display_progress.py -f " + self.dest_model + "/display_progress/display_data.pkl"
                display_proc = subprocess.Popen(command)
            elif platform.system() == 'Linux':
                command = "exec python3 display_progress.py -f " + self.dest_model + "/display_progress/display_data.pkl"
                display_proc = subprocess.Popen([command], shell=True)

        # open a session
        with tf.Session() as self.sess:

            # initialize the variables
            self.sess.run(tf.global_variables_initializer())

            # iterate
            epoch = 0
            t0 = time.time()

            # initial evaluation of the network
            fitted_p, metric_error, topo_error_in_P, topo_error_in_H, encoding, prediction, sensation = self.track_progress(data, disp)

            print("epoch: {:6d}, loss: _, metric error: {:.2e}, topo error in P: {:.2e}, topo error in H: {:.2e} - ({:.2f} sec)"
                  .format(epoch, metric_error, topo_error_in_P, topo_error_in_H, time.time() - t0))

            while epoch < n_epochs:

                # train for 1000 epochs
                epoch, current_loss = self.train(data=data, number_epochs=1000)

                # get tracked variables and send them to Tensorboard
                fitted_p, metric_error, topo_error_in_P, topo_error_in_H, encoding, prediction, sensation = self.track_progress(data, disp)

                print("epoch: {:6d}, loss: {:.2e}, metric error: {:.2e}, topo error in P: {:.2e}, topo error in H: {:.2e} - ({:.2f} sec)"
                      .format(epoch, current_loss, metric_error, topo_error_in_P, topo_error_in_H, time.time() - t0))

                # save the network
                self.save_network()

                if current_loss is None:
                    break

        # kill the display process
        if disp:
            display_proc.kill()

    def compute_weighted_affine_errors_in_P(self, target_set, origin_set, weight=0):
        """
        Compute the affine transformation: target_set = origin_set * coef_ + intercept_
        Estimate the error between the metrics of target_set and of the projection of origin_set in the target_set space.
        This error can be weighted to focus more or less on small distances in the target space.

        Inputs:
            target_set - (k, dim_target_space) array
            origin_set - (k, dim_origin_space) array
            weight - relative weight of smaller distances relative to large distances (weight >= 0, with weight = 0 for a uniform weighting)

        Returns:
            weighted_error - mean metric error between the projected set and the target_set
            fitted - linear projection of origin_set into the target space
        """

        # fit the linear regression
        self.lin_reg_model.fit(origin_set, target_set)

        # get the projection of origin_set into the target_set space
        fitted = self.lin_reg_model.predict(origin_set)

        # get the metrics of the target_set and the projection of origin_set
        pdist_target = pdist(target_set)
        pdist_fitted = pdist(fitted)

        # compute the mean weighted error between the metrics
        weighted_error = np.mean(np.absolute(pdist_fitted - pdist_target) / pdist_target.max() * np.exp(-weight * pdist_target / pdist_target.max()))

        return weighted_error, fitted

    def compute_topology_error_in_H(self, p_set, h_set, weight=10):
        """
        Estimates how much the topology of P_set is respected by H_set.

        Inputs:
            p_set - (k, dim_P) array
            h_set - (k, dim_H) array
            weight - relative weight of smaller P distances relative to large distances (weight >= 0, with weight = 0 for a uniform weighting)

        Returns:
            weighted_error - mean topological dissimilarity
        """

        # get the metrics of H_set and P_set
        pdist_h = pdist(h_set)
        pdist_p = pdist(p_set)

        # compute the mean weighted error between the metrics
        weighted_error = np.mean(pdist_h / pdist_h.max() * np.exp(-weight * pdist_p / pdist_p.max()))

        return weighted_error

    def track_progress(self, data, disp):
        """
        Computes and saves the variables tracked via Tensorboard + save the data to display by display_progress.py if disp=True.
        """

        # get the encoding of the regular motor sampling
        motor_encoding = self.sess.run(self.encode_module_t.output, feed_dict={self.motor_t: data["grid_motor"]})

        # compute the dissimilarities and affine projections
        metric_err, fitted_p = self.compute_weighted_affine_errors_in_P(data["grid_pos"], motor_encoding, weight=0)
        topo_err_in_P, _ = self.compute_weighted_affine_errors_in_P(data["grid_pos"], motor_encoding, weight=10)
        topo_err_in_H = self.compute_topology_error_in_H(data["grid_pos"], motor_encoding, weight=10)

        # get a random batch to evaluate the prediction error (without replace significantly increases computation time)
        batch_indexes = np.random.choice(data["motor_t"].shape[0], self.batch_size, replace=True)

        # perform sensory prediction and process the summaries
        curr_loss, curr_summaries, predicted_sensation, gt_sensation = self.sess.run([self.loss,
                                                                                     self.merged_summaries,
                                                                                     self.prediction_module.output,
                                                                                     self.sensor_tp],
                                                                                     feed_dict={self.motor_t: data["motor_t"][batch_indexes, :],
                                                                                                self.sensor_t: data["sensor_t"][batch_indexes, :],
                                                                                                self.motor_tp: data["motor_tp"][batch_indexes, :],
                                                                                                self.sensor_tp: data["sensor_tp"][batch_indexes, :],
                                                                                                self.metric_error: metric_err,
                                                                                                self.topology_error_in_P: topo_err_in_P,
                                                                                                self.topology_error_in_H: topo_err_in_H})

        # save the summaries
        curr_epoch = self.sess.run(self.global_step)
        self.summaries_writer.add_summary(curr_summaries, curr_epoch)

        if disp:
            # save the data to display by display_progress.py
            display_dict = {"epoch": curr_epoch,
                            "loss": curr_loss,
                            "motor": data["grid_motor"],
                            "gt_pos": data["grid_pos"],
                            "encoded_motor": motor_encoding,
                            "projected_encoding": fitted_p,
                            "metric_error": metric_err,
                            "topo_error_in_P": topo_err_in_P,
                            "topo_error_in_H": topo_err_in_H,
                            "gt_sensation": gt_sensation,
                            "predicted_sensation": predicted_sensation
                            }

            # write display_dict on the disk
            if not os.path.exists(self.dest_model + "/display_progress"):
                os.makedirs(self.dest_model + "/display_progress")
            with open(self.dest_model + "/display_progress/display_data.pkl", "wb") as file:
                pickle.dump(display_dict, file)

        return fitted_p, metric_err, topo_err_in_P, topo_err_in_H, motor_encoding, predicted_sensation, gt_sensation

    def save_network(self):
        """
        Saves the network in dir_model/model.
        """

        # destination where to save the model
        dest = self.dest_model + '/model'

        # create the folder if necessary
        if not os.path.exists(dest):
            os.makedirs(dest)

        # save the model
        self.saver.save(self.sess, dest + '/model.ckpt')

    def log(self, dest_log):
        """
        Writes the network's attributes to the disk.
        """

        serializable_dict = self.__dict__.copy()
        for key, value in self.__dict__.items():
            if type(value) is np.ndarray:
                serializable_dict[key] = value.tolist()
            elif type(value) not in (list, int, str):
                del(serializable_dict[key])

        with open(dest_log, "w") as file:
            json.dump(serializable_dict, file, indent=1)