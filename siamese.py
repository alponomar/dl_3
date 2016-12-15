from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import regularizers



class Siamese(object):
    """
    This class implements a siamese convolutional neural network in
    TensorFlow. Term siamese is used to refer to architectures which
    incorporate two branches of convolutional networks parametrized
    identically (i.e. weights are shared). These graphs accept two
    input tensors and a label in general.
    """

    def inference(self, x, reuse = False):
        """
        Defines the model used for inference. Output of this model is fed to the
        objective (or loss) function defined for the task.

        Here we recommend you to consider using variable and name scopes in order
        to make your graph more intelligible for later references in TensorBoard
        and so on. You can define a name scope for the whole model or for each
        operator group (e.g. conv+pool+relu) individually to group them by name.
        Variable scopes are essential components in TensorFlow for parameter sharing.
        You can use the variable scope to activate/deactivate 'variable reuse'.

        Args:
           x: 4D float Tensor of size [batch_size, input_height, input_width, input_channels]
           reuse: Python bool to switch reusing on/off.

        Returns:
           l2_out: L2-normalized output tensor of shape [batch_size, 192]

        Hint: Parameter reuse indicates whether the inference graph should use
        parameter sharing or not. You can study how to implement parameter sharing
        in TensorFlow from the following sources:

        https://www.tensorflow.org/versions/r0.11/how_tos/variable_scope/index.html
        """
        x_inp = tf.reshape(x, [-1, 32, 32, 3])
        with tf.variable_scope('ConvNet', reuse=reuse) as conv_scope:
            if reuse:
               conv_scope.reuse_variables()
            ########################
            # PUT YOUR CODE HERE  #
            ########################
            # raise NotImplementedError
            xavier = tf.contrib.layers.xavier_initializer()
            
            def _forward_conv_layer(name, w_shape, b_shape, x_inp, max_pool_kernel, max_pool_stride, act_func):
              with tf.variable_scope(name):
                W = tf.get_variable('W', w_shape, initializer=xavier)
                b = tf.get_variable('b', b_shape, initializer=tf.constant_initializer(0))
                conv_out = act_func(tf.nn.conv2d(x_inp, W, strides=[1, 1, 1, 1], padding='SAME') + b)
                out = tf.nn.max_pool(conv_out, ksize=[1, max_pool_kernel, max_pool_kernel, 1], strides=[1, max_pool_stride, max_pool_stride, 1], padding='SAME')
                tf.histogram_summary(name + '_weights', W)
                tf.histogram_summary(name + '_b', b)
                tf.histogram_summary(name + '_out', conv_out)
                tf.histogram_summary(name + '_maxpool', out)
                return out

            def _forward_fc_layer(name, w_shape, b_shape, x_inp, act_func):
              with tf.variable_scope(name):
                W = tf.get_variable('W', w_shape, initializer=xavier)
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

            flatten = tf.reshape(conv2, [-1, 8 * 8 * 64])

            self.fc1 = _forward_fc_layer(name='fc1', w_shape=[8 * 8 * 64, 384], b_shape=384, 
              x_inp=flatten, act_func=tf.nn.relu)
            self.fc2 = _forward_fc_layer(name='fc2', w_shape=[384, 192], b_shape=192, 
              x_inp=self.fc1, act_func=tf.nn.relu)
            with tf.variable_scope('l2_norm'):
              self.l2_out = tf.nn.l2_normalize(self.fc2, 1)
              tf.histogram_summary('l2_out', self.l2_out)
            """
            with tf.variable_scope('siamese') as conv_scope:
	            ########################
	            # PUT YOUR CODE HERE  #
	            ########################
	            xavier = tf.contrib.layers.xavier_initializer()
	            l2_reg = tf.contrib.layers.l2_regularizer(0.)
	            filter_depth = 3
	            batch_size = tf.shape(x)[0]
	            # Compute the two convolutional layers in a loop
	            for i in range(2):
	                with tf.variable_scope('conv' + str(i+1), reuse=reuse):
	                    # Get the 5x5 filters with 3 color channels and 64 output channels
	                    filters = tf.get_variable('filter',
	                                              shape=[5, 5, filter_depth, 64],
	                                              initializer=xavier,
	                                              dtype=tf.float32)
	                    tf.histogram_summary('siamese' + str(channel) + '/conv' + str(i+1) + '/filters',
	                                         filters)
	                    # Do the convolution
	                    x = tf.nn.conv2d(x,
	                                     filters,
	                                     [1, 1, 1, 1],
	                                     "SAME",
	                                     name='convolve')
	                    # Activation function
	                    x = tf.nn.relu(x, name='ReLU')
	                    # Max pooling
	                    x = tf.nn.max_pool(x,
	                                       [1, 3, 3, 1],
	                                       [1, 2, 2, 1],
	                                       'SAME',
	                                       name='pooling')
	                    filter_depth = 64
	            # Flatten the tensor
	            x = tf.reshape(x, [batch_size, -1], name='flatten')
	            # Compute the three fully connected layers in for loop
	            input_size = 4096
	            output_size = 384
	            for i in range(2):
	                with tf.variable_scope('fc' + str(i+1), reuse=reuse):
	                    # Get the weights
	                    weights = tf.get_variable('weights',
	                                              shape=[input_size, output_size],
	                                              initializer=xavier,
	                                              regularizer=l2_reg,
	                                              dtype=tf.float32)
	                    tf.histogram_summary('siamese' + str(channel) + '/fc' + str(i+1) + '/weights',
	                                         weights)
	                    # Get the bias
	                    bias = tf.get_variable('bias',
	                                           shape=[output_size,],
	                                           initializer=tf.constant_initializer(0.),
	                                           dtype=tf.float32)
	                    tf.histogram_summary('siamese' + str(channel) + '/fc' + str(i+1) + '/bias',
	                                         bias)
	                    # Compute the output
	                    x = tf.nn.relu(tf.add(tf.matmul(x, weights), bias),
	                                   name='siamese' + str(channel) + '/fc' + str(i+1) + '/fc' + str(i+1))
	                    input_size = output_size
	                    output_size /= 2
	            # normalise with the 2-norm
	            l2_out = tf.nn.l2_normalize(x, [1], name='l2_norm')
	            tf.histogram_summary('siamese' + str(channel) + '/l2_out', l2_out)
	        """
            ########################
            # END OF YOUR CODE    #
            ########################
        return self.l2_out

    def loss(self, channel_1, channel_2, label, margin):
        """
        Defines the contrastive loss. This loss ties the outputs of
        the branches to compute the following:

               L =  Y * d^2 + (1-Y) * max(margin - d^2, 0)

               where d is the L2 distance between the given
               input pair s.t. d = ||x_1 - x_2||_2 and Y is
               label associated with the pair of input tensors.
               Y is 1 if the inputs belong to the same class in
               CIFAR10 and is 0 otherwise.

               For more information please see:
               http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

        Args:
            channel_1: output of first channel (i.e. branch_1),
                              tensor of size [batch_size, 192]
            channel_2: output of second channel (i.e. branch_2),
                              tensor of size [batch_size, 192]
            label: Tensor of shape [batch_size]
            margin: Margin of the contrastive loss

        Returns:
            loss: scalar float Tensor
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################
        """          
        d2 = tf.reduce_sum(tf.square(channel_1 - channel_2))
        contrastive_loss_all = label * d2 + (1. - label) * tf.maximum(margin - d2, 0.)

        loss = tf.reduce_mean(contrastive_loss_all)
        """
        # layers_reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        # loss = layers_reg_loss + contrastive_loss

        # tf.scalar_summary('reg loss', layers_reg_loss)
        # tf.scalar_summary('contrastive loss', contrastive_loss)
         
        distance_squared = tf.reduce_sum(tf.square(tf.sub(channel_1,
                                                          channel_2)), 1)
        loss = tf.mul(label, distance_squared) + \
               tf.mul(tf.sub(1., label),
                      tf.square(tf.maximum(tf.sub(margin,
                                                  tf.sqrt(tf.add(distance_squared,
                                                                 1e-6))), 0.)))
        loss = tf.reduce_mean(loss)
        
        tf.scalar_summary('contrastive loss', loss)
        # raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        ########################

        return loss
