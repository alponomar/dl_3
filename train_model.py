from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from sklearn.metrics import accuracy_score

import tensorflow as tf
import numpy as np
import cifar10_utils
from cifar10_siamese_utils import get_cifar10 as get_cifar_10_siamese
from cifar10_siamese_utils import create_dataset as create_dataset_siamese
from convnet import ConvNet
from siamese import Siamese
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import pandas as pd
from matplotlib.pyplot import cm 

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'



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
    Performs training and evaluation of ConvNet model.

    First define your graph using class ConvNet and its methods. Then define
    necessary operations such as trainer (train_step in this case), savers
    and summarizers. Finally, initialize your model within a tf.Session and
    do the training.

    ---------------------------
    How to evaluate your model:
    ---------------------------
    Evaluation on test set should be conducted over full batch, i.e. 10k images,
    while it is alright to do it over minibatch for train set.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every print_freq iterations
    - on test set every eval_freq iterations

    ------------------------
    Additional requirements:
    ------------------------
    Also you are supposed to take snapshots of your model state (i.e. graph,
    weights and etc.) every checkpoint_freq iterations. For this, you should
    study TensorFlow's tf.train.Saver class. For more information, please
    checkout:
    [https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html]
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


    cnn = ConvNet()

    x = tf.placeholder(tf.float32, shape=(None, input_data_dim, input_data_dim, 3), name="x")
    y = tf.placeholder(tf.float32, shape=(None, n_classes), name="y")

    with tf.name_scope('train_cnn'):
        infs = cnn.inference(x)
        with tf.name_scope('cross-entropy-loss'): 
          loss = cnn.loss(infs, y)
        with tf.name_scope('accuracy'): 
          accuracy = cnn.accuracy(infs, y)
        fc2 = cnn.fc2
        merged = tf.merge_all_summaries()
        opt_operation = train_step(loss)

    
    with tf.Session() as sess:
        saver = tf.train.Saver() 
    
        sess.run(tf.initialize_all_variables())

        test_acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
        print("Initial Test Accuracy = {0:.3f}".format(test_acc))

        train_writer = tf.train.SummaryWriter(FLAGS.log_dir + "/train/", sess.graph)
        test_writer = tf.train.SummaryWriter(FLAGS.log_dir + "/test/", sess.graph)

        for iteration in range(FLAGS.max_steps + 1):
            x_batch, y_batch = cifar10.train.next_batch(FLAGS.batch_size)      
            _ = sess.run([opt_operation], feed_dict={x: x_batch, y: y_batch})

            if iteration % FLAGS.print_freq == 0:
                [train_acc, train_loss, summary_train] = sess.run([accuracy, loss, merged], feed_dict={x: x_batch, y: y_batch})
                train_writer.add_summary(summary_train, iteration)
                print("Iteration {0:d}/{1:d}. Train Loss = {2:.3f}, Train Accuracy = {3:.3f}".
                            format(iteration, FLAGS.max_steps, train_loss, train_acc))


            if iteration % FLAGS.eval_freq == 0:
                [test_acc, test_loss, summary_test] = sess.run([accuracy, loss, merged], feed_dict={x: x_test, y: y_test})
                test_writer.add_summary(summary_test, iteration)
                print("Iteration {0:d}/{1:d}. Test Loss = {2:.3f}, Test Accuracy = {3:.3f}".
                                format(iteration, FLAGS.max_steps, test_loss, test_acc))

            if iteration > 0 and iteration % FLAGS.checkpoint_freq == 0:
                saver.save(sess, FLAGS.checkpoint_dir + '/cnn_model.ckpt')
        
        train_writer.flush()
        test_writer.flush()
        train_writer.close()
        test_writer.close()

        test_acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
        print("Final Test Accuracy = {0:.3f}".format(test_acc))

        sess.close()  

    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    ########################


def train_siamese():
    """
    Performs training and evaluation of Siamese model.

    First define your graph using class Siamese and its methods. Then define
    necessary operations such as trainer (train_step in this case), savers
    and summarizers. Finally, initialize your model within a tf.Session and
    do the training.

    ---------------------------
    How to evaluate your model:
    ---------------------------
    On train set, it is fine to monitor loss over minibatches. On the other
    hand, in order to evaluate on test set you will need to create a fixed
    validation set using the data sampling function you implement for siamese
    architecture. What you need to do is to iterate over all minibatches in
    the validation set and calculate the average loss over all minibatches.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every print_freq iterations
    - on test set every eval_freq iterations

    ------------------------
    Additional requirements:
    ------------------------
    Also you are supposed to take snapshots of your model state (i.e. graph,
    weights and etc.) every checkpoint_freq iterations. For this, you should
    study TensorFlow's tf.train.Saver class. For more information, please
    checkout:
    [https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html]
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.

    def _check_loss(data):
        loss_val = 0.
        for batch in data:
            x1_data, x2_data, y_data = batch
            [curr_loss] = sess.run([loss], feed_dict={x1: x1_data, x2: x2_data, y: y_data})
            loss_val += curr_loss
        return loss_val / len(data)

    tf.set_random_seed(42)
    np.random.seed(42)
    cifar10 = get_cifar_10_siamese(FLAGS.data_dir, validation_size=5000)
    val_data  = create_dataset_siamese(cifar10.validation, num_tuples = 1000, fraction_same = 0.2)
    test_data  = create_dataset_siamese(cifar10.test, num_tuples = 1000, fraction_same = 0.2)
   
    #### PARAMETERS
    classes = ['plane', 'car', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']
    n_classes = len(classes)
    input_data_dim = cifar10.test.images.shape[1]
    #####

    cnn_siamese = Siamese()

    x1 = tf.placeholder(tf.float32, shape=(None, input_data_dim, input_data_dim, 3), name="x1")
    x2 = tf.placeholder(tf.float32, shape=(None, input_data_dim, input_data_dim, 3), name="x2")
    y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

    with tf.name_scope('train_cnn'):
        infs1 = cnn_siamese.inference(x1, reuse=None)
        infs2 = cnn_siamese.inference(x2, reuse=True)
        with tf.name_scope('cross-entropy-loss'): 
          loss = cnn_siamese.loss(infs1, infs2, y, 0.1)
        merged = tf.merge_all_summaries()
        opt_operation = train_step(loss)
    
    with tf.Session() as sess:

        saver = tf.train.Saver() 

        sess.run(tf.initialize_all_variables())
        # print("testing!")
        test_loss = _check_loss(test_data)
        print("Initial Test Loss = {0:.3f}".format(test_loss))

        # train_writer = tf.train.SummaryWriter(FLAGS.log_dir + "/train/", sess.graph)
        # test_writer = tf.train.SummaryWriter(FLAGS.log_dir + "/test/", sess.graph)
        val_losses = []
        train_losses = []
        for iteration in range(FLAGS.max_steps + 1):
          # print(iteration)
            x1_train, x2_train, y_train = cifar10.train.next_batch(FLAGS.batch_size)
            _ = sess.run([opt_operation], feed_dict={x1: x1_train, x2: x2_train, y: y_train})

            if iteration % FLAGS.print_freq == 0:
                [train_loss] = sess.run([loss], feed_dict={x1: x1_train, x2: x2_train, y: y_train})
                train_losses.append(train_loss)
                # train_writer.add_summary(summary_train, iteration)
                print("Iteration {0:d}/{1:d}. Train Loss = {2:.3f}".
                                format(iteration, FLAGS.max_steps, train_loss))
               
            if iteration % FLAGS.eval_freq == 0:
                val_loss = _check_loss(val_data)
                val_losses.append(val_loss)
                # [test_acc, test_loss, summary_test] = sess.run([accuracy, loss, merged], feed_dict={x: x_test, y: y_test})
                # test_writer.add_summary(summary_test, iteration)
                print("Iteration {0:d}/{1:d}. Validation Loss = {2:.3f}".
                                format(iteration, FLAGS.max_steps, val_loss))

            if iteration > 0 and iteration % FLAGS.checkpoint_freq == 0:
                saver.save(sess, FLAGS.checkpoint_dir + '/cnn_model_siamese.ckpt')

        test_loss = _check_loss(test_data)
        print("Final Test Loss = {0:.3f}".format(test_loss))
        # train_writer.flush()
        # test_writer.flush()
        # train_writer.close()
        # test_writer.close()

        sess.close() 
    print("train_loss", train_losses)
    print("val_loss", val_losses)
    #######################
    # PUT YOUR CODE HERE  #
    ########################

    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    ########################

def _get_conf_mat(test_predictions, true_labels, name, classes):
  """
  Constructs confusion matrix
  Args:
      test_predictions: predicted labels
      true_labels: ground truth
      classes: name of classes
  """
  ground_truth = pd.Series(true_labels, name='True')
  predictions = pd.Series(test_predictions, name='Predicted')
  conf_mat = pd.crosstab(ground_truth, predictions)

  plt.figure()  
  plt.xticks(range(0, 100, 1), classes)
  plt.yticks(range(0, 100, 1), classes)
  plt.title(name + ', confusion matrix', fontsize = 20)
  data = conf_mat.values / np.amax(conf_mat.values) * 255.
  plt.imshow(data, interpolation='none')
  plt.savefig(name + '_conf_mat.png')
  # plt.show()
  predicted_classes = [classes[i] for i in range(len(classes)) if i in list(test_predictions)]

  conf_mat.columns = predicted_classes
  conf_mat['True'] = classes
  cols = conf_mat.columns.tolist()
  cols = cols[-1:] + cols[:-1]
  conf_mat = conf_mat[cols]  
  print(conf_mat)

def _train_one_vs_all(features, y_test, name, classes):
    prop = 0.8
    x_train_1vsall = features[0:len(features)*prop]
    y_train_1vsall = y_test[0:len(y_test)*prop]

    x_test_1vsall= features[len(features)*prop:len(features)]
    y_test_1vsall = y_test[len(y_test)*prop:len(y_test)]
    model = OneVsRestClassifier(LinearSVC(random_state=0)).fit(x_train_1vsall, y_train_1vsall)
    predictions = model.predict(x_test_1vsall)
    print(name + " accuracy:" , accuracy_score(predictions, y_test_1vsall))
    _get_conf_mat(predictions, y_test_1vsall, name, classes)


def _plot_tsne(name, features, y):
    classes = ['plane', 'car', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']
    colors = ['r','g','b','y', 'm', 'c', 'w', 'k', 'chartreuse', 'gray']
    markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']

    model = TSNE(random_state=0, perplexity=30, verbose=10).fit_transform(features)
    labels = [classes[lab] for lab in y]
    plt.figure()
    for class_id in range(len(classes)):
        model_x = [model[i, 0] for i in range(len(y)) if y[i] == class_id]
        model_y = [model[i, 1] for i in range(len(y)) if y[i] == class_id]
        plt.scatter(model_x, model_y, c=colors[class_id], label = classes[class_id], marker = markers[class_id])
    plt.legend(loc='upper center', ncol=5, prop={'size':9})
    plt.savefig(name)

def feature_extraction():
    """
    This method restores a TensorFlow checkpoint file (.ckpt) and rebuilds inference
    model with restored parameters. From then on you can basically use that model in
    any way you want, for instance, feature extraction, finetuning or as a submodule
    of a larger architecture. However, this method should extract features from a
    specified layer and store them in data files such as '.h5', '.npy'/'.npz'
    depending on your preference. You will use those files later in the assignment.

    Args:
        [optional]
    Returns:
        None
    """

    ########################
    # PUT YOUR CODE HERE  #
    ########################

    
    tf.reset_default_graph()
    classes = ['plane', 'car', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']
    
    tf.set_random_seed(42)
    np.random.seed(42)
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    x_test, y_test = cifar10.test.images, cifar10.test.labels
    y_test = np.argmax(y_test, axis=1)
    input_data_dim = cifar10.test.images.shape[1]
    n_classes = 10
    cnn = ConvNet()
  
    x = tf.placeholder(tf.float32, shape=(None, input_data_dim, input_data_dim, 3), name="x")
    y = tf.placeholder(tf.float32, shape=(None, n_classes), name="y")

    with tf.name_scope('train_cnn'):
        infs = cnn.inference(x)
        flatten = cnn.flatten
        fc1 = cnn.fc1
        fc2 = cnn.fc2


    with tf.Session() as sess:

        saver = tf.train.Saver()
        
        saver.restore(sess, FLAGS.checkpoint_dir + '/cnn_model.ckpt')
        
        fc2_features = sess.run([fc2], feed_dict={x: x_test})[0]
       
        _plot_tsne("fc2.png", fc2_features, y_test)

        fc1_features = sess.run([fc1], feed_dict={x: x_test})[0]
        _plot_tsne("fc1.png",  fc1_features, y_test)

        flatten_features = sess.run([flatten], feed_dict={x: x_test})[0]
        _plot_tsne("flatten.png", flatten_features, y_test)
       
    _train_one_vs_all(fc2_features, y_test, "FC2", classes)
    _train_one_vs_all(fc1_features, y_test, "FC1", classes)
    _train_one_vs_all(flatten_features, y_test, "Flatten", classes)
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    ########################

def feature_extraction_siamese():
    """
    This method restores a TensorFlow checkpoint file (.ckpt) and rebuilds inference
    model with restored parameters. From then on you can basically use that model in
    any way you want, for instance, feature extraction, finetuning or as a submodule
    of a larger architecture. However, this method should extract features from a
    specified layer and store them in data files such as '.h5', '.npy'/'.npz'
    depending on your preference. You will use those files later in the assignment.

    Args:
        [optional]
    Returns:
        None
    """

    ########################
    # PUT YOUR CODE HERE  #
    ########################

    tf.reset_default_graph()

    classes = ['plane', 'car', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']
    tf.set_random_seed(42)
    np.random.seed(42)
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    x_test, y_test = cifar10.test.images, cifar10.test.labels
    y_test = np.argmax(y_test, axis=1)
    input_data_dim = cifar10.test.images.shape[1]
    n_classes = 10

    cnn_siamese = Siamese()

    x = tf.placeholder(tf.float32, shape=(None, input_data_dim, input_data_dim, 3), name="x1")
    y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

    with tf.name_scope('train_cnn'):
        infs1 = cnn_siamese.inference(x, reuse=None)
        l2_out = cnn_siamese.l2_out


    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.checkpoint_dir + '/cnn_model_siamese.ckpt')

        l2_out_features = sess.run([l2_out], feed_dict={x: x_test})[0]
        _plot_tsne("L2 out", l2_out_features, y_test)
        _train_one_vs_all(l2_out_features, y_test, "L2 norm", classes)        

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
    
    if str(FLAGS.is_train) == "True":
        if FLAGS.train_model == 'linear':
            train()
        elif FLAGS.train_model == 'siamese':
            train_siamese()
        else:
            raise ValueError("--train_model argument can be linear or siamese")
    else:
        if FLAGS.train_model == 'linear':
            feature_extraction()
        elif FLAGS.train_model == 'siamese':
            feature_extraction_siamese()
        else:
            raise ValueError("--train_model argument can be linear or siamese")

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
    parser.add_argument('--checkpoint_freq', type = int, default = CHECKPOINT_FREQ_DEFAULT,
                      help='Frequency with which the model state is saved.')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
    parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
    parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
                      help='Checkpoint directory')
    parser.add_argument('--is_train', type = str, default = "True",
                      help='Training or feature extraction')
    parser.add_argument('--train_model', type = str, default = 'linear',
                      help='Type of model. Possible options: linear and siamese')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
