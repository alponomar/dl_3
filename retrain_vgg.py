from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np
from vgg import load_pretrained_VGG16_pool5
import cifar10_utils
from fc import FC

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'
REFINE_AFTER_K_STEPS_DEFAULT = 0

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cifar10'
CHECKPOINT_DIR_DEFAULT = './checkpoints'

def train_step(loss):
    """
    Defines the ops to conduct an optimization step. You can set a learning
    rate scheduler or pick your favorite optimizer here. This set of operations
    should be applicable to both ConvNet() and Siamese() objects.

    Args:
        loss: scalar float Tensor, full loss = cross_entropy + reg_loss

    Returns:
        train_op: Ops for optimization.
    """
    ########################
    # PUT YOUR CODE HERE  #
    ########################
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    optimizer.learning_rate = FLAGS.learning_rate
    train_op = optimizer.minimize(loss)
    ########################
    # END OF YOUR CODE    #
    ########################

    return train_op

def train():
    """
    Performs training and evaluation of your model.

    First define your graph using vgg.py with your fully connected layer.
    Then define necessary operations such as trainer (train_step in this case),
    savers and summarizers. Finally, initialize your model within a
    tf.Session and do the training.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every PRINT_FREQ iterations
    - on test set every EVAL_FREQ iterations

    ---------------------------
    How to evaluate your model:
    ---------------------------
    Evaluation on test set should be conducted over full batch, i.e. 10k images,
    while it is alright to do it over minibatch for train set.
    """
    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    x_test, y_test = cifar10.test.images, cifar10.test.labels
   
    #### PARAMETERS
    classes = ['plane', 'car', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']
    n_classes = len(classes)
    input_data_dim = cifar10.test.images.shape[1]
    #####

    
    fc = FC()

    refine_after_k = tf.placeholder(tf.bool)
    x = tf.placeholder(tf.float32, shape=(None, None, None, 3), name="x")
    y = tf.placeholder(tf.float32, shape=(None, n_classes), name="y")

    with tf.name_scope('refine_cnn'):
        pool5, assign_ops = load_pretrained_VGG16_pool5(x)
        pool5 = tf.cond(refine_after_k, lambda: tf.stop_gradient(pool5), lambda: pool5)

        infs = fc.inference(pool5)
        with tf.name_scope('cross-entropy-loss'): 
          loss = fc.loss(infs, y)
        with tf.name_scope('accuracy'): 
          accuracy = fc.accuracy(infs, y)

        merged = tf.merge_all_summaries()
        opt_operation = train_step(loss)

    
    with tf.Session() as sess:
        saver = tf.train.Saver() 
    
        sess.run(tf.initialize_all_variables())
        for op in assign_ops:
            sess.run(op)

        test_acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test, refine_after_k: False})
        print("Initial Test Accuracy = {0:.3f}".format(test_acc))

        train_writer = tf.train.SummaryWriter(FLAGS.log_dir + "/train/", sess.graph)
        test_writer = tf.train.SummaryWriter(FLAGS.log_dir + "/test/", sess.graph)

        for iteration in range(FLAGS.max_steps + 1):
            x_batch, y_batch = cifar10.train.next_batch(FLAGS.batch_size)      
            _ = sess.run([opt_operation], feed_dict={x: x_batch, y: y_batch, refine_after_k: FLAGS.refine_after_k > iteration})

            if iteration % FLAGS.print_freq == 0:
                [train_acc, train_loss, summary_train] = sess.run([accuracy, loss, merged], feed_dict={x: x_batch, y: y_batch, refine_after_k: False})
                train_writer.add_summary(summary_train, iteration)
                print("Iteration {0:d}/{1:d}. Train Loss = {2:.3f}, Train Accuracy = {3:.3f}".
                                format(iteration, FLAGS.max_steps, train_loss, train_acc))
               

            if iteration % FLAGS.eval_freq == 0:
                [test_acc, test_loss, summary_test] = sess.run([accuracy, loss, merged], feed_dict={x: x_test, y: y_test, refine_after_k:False})
                test_writer.add_summary(summary_test, iteration)
                print("Iteration {0:d}/{1:d}. Test Loss = {2:.3f}, Test Accuracy = {3:.3f}".
                                format(iteration, FLAGS.max_steps, test_loss, test_acc))

            if iteration > 0 and iteration % FLAGS.checkpoint_freq == 0:
                saver.save(sess, FLAGS.checkpoint_dir + '/vgg_model.ckpt')
        
        train_writer.flush()
        test_writer.flush()
        train_writer.close()
        test_writer.close()

        test_acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test, refine_after_k: False})
        print("Final Test Accuracy = {0:.3f}".format(test_acc))

        sess.close()
     

    # raise NotImplementedError

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    ########################

def initialize_folders():
    """
    Initializes all folders in FLAGS variable.
    """

    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)

    if not tf.gfile.Exists(FLAGS.data_dir):
        tf.gfile.MakeDirs(FLAGS.data_dir)

    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main(_):
    print_flags()

    initialize_folders()
    train()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
    parser.add_argument('--print_freq', type = int, default = PRINT_FREQ_DEFAULT,
                      help='Frequency of evaluation on the train set')
    parser.add_argument('--eval_freq', type = int, default = EVAL_FREQ_DEFAULT,
                      help='Frequency of evaluation on the test set')
    parser.add_argument('--refine_after_k', type = int, default = REFINE_AFTER_K_STEPS_DEFAULT,
                      help='Number of steps after which to refine VGG model parameters (default 0).')
    parser.add_argument('--checkpoint_freq', type = int, default = CHECKPOINT_FREQ_DEFAULT,
                      help='Frequency with which the model state is saved.')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
    parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
    parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
                      help='Checkpoint directory')


    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
