from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import regularizers


class ConvNet(object):
    """
   This class implements a convolutional neural network in TensorFlow
   It incorporates a certain graph model to be trained and to be used
   in inference
    """

    def __init__(self, n_classes = 10):
        """
        Constructor for an ConvNet object Default values should be used as hints for
        the usage of each parameter
        Args:
          n_classes: int, number of classes of the classification problem
                          This number is required in order to specify the
                          output dimensions of the ConvNet
        """
        self.n_classes = n_classes

    def inference(self, x):
        """
        Performs inference given an input tensor This is the central portion
        of the network where we describe the computation graph Here an input
        tensor undergoes a series of convolution, pooling and nonlinear operations
        as defined in this method For the details of the model, please
        see assignment file

        Here we recommend you to consider using variable and name scopes in order
        to make your graph more intelligible for later references in TensorBoard
        and so on You can define a name scope for the whole model or for each
        operator group (eg conv+pool+relu) individually to group them by name
        Variable scopes are essential components in TensorFlow for parameter sharing
        Although the model(s) which are within the scope of this class do not require
        parameter sharing it is a good practice to use variable scope to encapsulate
        model

        Args:
          x: 4D float Tensor of size [batch_size, input_height, input_width, input_channels]

        Returns:
          logits: 2D float Tensor of size [batch_size, self.n_classes] Returns
                  the logits outputs (before softmax transformation) of the
                  network These logits can then be used with loss and accuracy
                  to evaluate the model
        """
        x_inp = tf.reshape(x, [-1, 32, 32, 3])
        with tf.variable_scope('ConvNet'):

            def _forward_conv_layer(name, w_shape, b_shape, x_inp, max_pool_kernel, max_pool_stride, act_func):
              with tf.variable_scope(name):
                W = tf.get_variable('W', w_shape, initializer=tf.random_normal_initializer(mean = 0.0, stddev=1e-3, dtype=tf.float32))
                b = tf.get_variable('b', b_shape, initializer=tf.constant_initializer(0))
                conv_out = act_func(tf.nn.conv2d(x_inp, W, strides=[1, 1, 1, 1], padding='SAME') + b)
                out = tf.nn.max_pool(conv_out, ksize=[1, max_pool_kernel, max_pool_kernel, 1], strides=[1, max_pool_stride, max_pool_stride, 1], padding='SAME')
                tf.histogram_summary(name + '_weights', W)
                tf.histogram_summary(name + '_b', b)
                tf.histogram_summary(name + '_out', conv_out)
                tf.histogram_summary(name + '_maxpool', out)
                return out

            def _forward_fc_layer(name, w_shape, b_shape, x_inp, regularizer_strength, act_func):
              with tf.variable_scope(name):
                W = tf.get_variable('W', w_shape, initializer=tf.random_normal_initializer(mean = 0.0, stddev=1e-3, dtype=tf.float32),regularizer = regularizers.l2_regularizer(regularizer_strength))
                b = tf.get_variable('b', b_shape, initializer=tf.constant_initializer(0))
                out = act_func(tf.matmul(x_inp, W) + b)
                tf.histogram_summary(name + '_weights', W)
                tf.histogram_summary(name + '_b', b)
                tf.histogram_summary(name + '_out', out)
              return out

            conv1 = _forward_conv_layer(name='conv1', w_shape=[5, 5, 3, 64], b_shape=64, 
              x_inp=x_inp, max_pool_kernel=3, max_pool_stride=2, act_func=tf.nn.relu)  
            conv2 = _forward_conv_layer(name='conv2', w_shape=[5, 5, 64, 64], b_shape=64, 
              x_inp=conv1, max_pool_kernel=3, max_pool_stride=2, act_func=tf.nn.relu) 

            self.flatten = tf.reshape(conv2, [-1, 8 * 8 * 64])

            self.fc1 = _forward_fc_layer(name='fc1', w_shape=[8 * 8 * 64, 384], b_shape=384, 
              x_inp=self.flatten, regularizer_strength=0.001, act_func=tf.nn.relu)
            self.fc2 = _forward_fc_layer(name='fc2', w_shape=[384, 192], b_shape=192, 
              x_inp=self.fc1, regularizer_strength=0.001, act_func=tf.nn.relu)
            logits = _forward_fc_layer(name='logits', w_shape=[192, 10], b_shape=10, 
              x_inp=self.fc2, regularizer_strength=0.001, act_func=lambda x: x)


        return logits

    def accuracy(self, logits, labels):
        """
        Calculate the prediction accuracy, ie the average correct predictions
        of the network
        As in self.loss above, you can use tf.scalar_summary to save
        scalar summaries of accuracy for later use with the TensorBoard

        Args:
          logits: 2D float Tensor of size [batch_size, self.n_classes]
                       The predictions returned through self.inference
          labels: 2D int Tensor of size [batch_size, self.n_classes]
                     with one-hot encoding Ground truth labels for
                     each observation in batch

        Returns:
          accuracy: scalar float Tensor, the accuracy of predictions,
                    ie the average correct predictions over the whole batch
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################
        correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', accuracy)
        # raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        ########################

        return accuracy

    def loss(self, logits, labels):
        """
        Calculates the multiclass cross-entropy loss from the logits predictions and
        the ground truth labels The function will also add the regularization
        loss from network weights to the total loss that is return
        In order to implement this function you should have a look at
        tf.nn.softmax_cross_entropy_with_logits
        You can use tf.scalar_summary to save scalar summaries of
        cross-entropy loss, regularization loss, and full loss (both summed)
        for use with TensorBoard This will be useful for compiling your report

        Args:
          logits: 2D float Tensor of size [batch_size, self.n_classes]
                       The predictions returned through self.inference
          labels: 2D int Tensor of size [batch_size, self.n_classes]
                       with one-hot encoding Ground truth labels for each
                       observation in batch

        Returns:
          loss: scalar float Tensor, full loss = cross_entropy + reg_loss
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################
	layers_reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='crossentropy')  
        ce_loss = tf.reduce_mean(cross_entropy, name='ce_loss')
	loss = layers_reg_loss + ce_loss
        tf.scalar_summary('cross-entropy loss', ce_loss)	
	tf.scalar_summary('regularization loss', layers_reg_loss)
	tf.scalar_summary('loss', loss)
        # loss = cross_entropy
        # raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        ########################

        return loss
