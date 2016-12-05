cnn = convnet()

x = tf.placeholder(tf.float32, shape=(None, input_data_dim), name="x")
y = tf.placeholder(tf.int32, shape=(None, n_classes), name="y")

with tf.name_scope('train'):
    infs = convnet.inference(x)
    with tf.name_scope('cross-entropy-loss'): 
      loss = convnet.loss(infs, y)
    with tf.name_scope('accuracy'): 
      accuracy = convnet.accuracy(infs, y)
    merged = tf.merge_all_summaries()
    optimizer = OPTIMIZER_DICT[(FLAGS.optimizer)](learning_rate)
    optimizer.learning_rate = learning_rate
    opt_operation = optimizer.minimize(loss)