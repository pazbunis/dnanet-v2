import tensorflow as tf
import numpy as np
from os import listdir, path
from enhancersdata import EnhancersData
from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment()
ex.observers.append(FileStorageObserver.create('log'))


@ex.config
def general_config():
    general_cfg = {
                    "seq_length": 1000,
                    "num_outs": 2,
                    "batch_size": 100,
                    "num_epochs": 40
                    }

@ex.config
def cnn_config():
    conv1_cfg = {"num_filters": 16, "filter_size": [4, 15]}
    conv2_cfg = {"num_filters": 16, "filter_size": [1, 5]}
    conv3_cfg = {"num_filters": 32, "filter_size": [1, 5]}
    pool1_cfg = {"kernel_size": 10, "stride": 10}
    pool2_cfg = {"kernel_size": 3, "stride": 3}
    pool3_cfg = {"kernel_size": 3, "stride": 3}
    dropout_keep_prob = 0.5

# model
@ex.capture
def CNN(x, general_cfg, conv1_cfg, conv2_cfg, conv3_cfg, pool1_cfg, pool2_cfg, pool3_cfg, dropout_keep_prob):
    x_seq = tf.reshape(x, [-1, 4, 1000, 1])
    conv1 = tf.layers.conv2d(inputs=x_seq, filters=conv1_cfg["num_filters"], kernel_size=conv1_cfg["filter_size"],
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv1")

    max_pool1 = tf.nn.max_pool(conv1, ksize=[1, 1, pool1_cfg["kernel_size"], 1],
                               strides=[1, 1, pool1_cfg["stride"], 1], padding='VALID')
    # conv2
    conv2 = tf.layers.conv2d(inputs=max_pool1, filters=conv2_cfg["num_filters"], kernel_size=conv2_cfg["filter_size"],
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv2")
    max_pool2 = tf.nn.max_pool(conv2, ksize=[1, 1, pool2_cfg["kernel_size"], 1],
                               strides=[1, 1, pool2_cfg["stride"], 1], padding='VALID')
    # conv3
    conv3 = tf.layers.conv2d(inputs=max_pool2, filters=conv3_cfg["num_filters"], kernel_size=conv3_cfg["filter_size"],
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv3")
    max_pool3 = tf.nn.max_pool(conv3, ksize=[1, 1, pool3_cfg["kernel_size"], 1],
                                      strides=[1, 1, pool3_cfg["stride"], 1], padding='VALID')
    # flatten
    conv3_flat = tf.contrib.layers.flatten(max_pool3)
    # two affine (fully-connected) layers with dropout in between
    dropout = tf.nn.dropout(conv3_flat, keep_prob=dropout_keep_prob)

    aff1 = tf.layers.dense(dropout, 100, activation=tf.nn.relu, name="affine1")
    #        aff2 = tf.layers.dense(aff1, , activation=tf.nn.relu, name="affine2")
    # output layer
    aff_out = tf.layers.dense(aff1, general_cfg["num_outs"], activation=None, name="affine_out")
    # we're returning the unscaled output so we can use the safe: tf.nn.softmax_cross_entropy_with_logits
    return aff_out


def log_files(dir_path):
    for filename in listdir(dir_path):
        ex.add_resource(path.join(dir_path, filename))

@ex.automain
def run_experiment(general_cfg, seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    data_dir_path = '/cs/grad/pazbu/paz/dev/projects/dnanet-v2/data'
    ds = EnhancersData(data_dir_path)
    log_files(data_dir_path)

    x = tf.placeholder(tf.float32, shape=[None, 4, 1000])
    y_ = tf.placeholder(tf.uint8, shape=[None, 2])
    keep_prob = tf.placeholder(tf.float32)

    y_conv = CNN(x, dropout_keep_prob=keep_prob)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    train_step = tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)

    y_pred_sig = tf.sigmoid(y_conv)
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_pred_sig, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    num_runs = 2
    num_epochs = general_cfg["num_epochs"]
    mini_batch_size = general_cfg["batch_size"]
    iters_per_epoch = int(ds.train.num_examples / mini_batch_size)
    
    with tf.Session() as sess:
        print('run, epoch, step, training accuracy, validation accuracy')
        for run_idx in range(num_runs):
            sess.run(tf.global_variables_initializer())
            for epoch_idx in range(num_epochs):
                for iter_idx in range(iters_per_epoch):
                    batch = ds.train.next_batch(mini_batch_size)
                    if iter_idx % 100 == 0:
                        # out = y_.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                        # print(out)
                        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                        valid_accuracy = accuracy.eval(feed_dict={x: ds.validation.seqs,
                                                                  y_: ds.validation.labels,
                                                                  keep_prob: 1.0})
                        print('run: %d, epoch: %d, iteration: %d, train accuracy: %g, validation accuracy: %g' %
                              (run_idx, epoch_idx, iter_idx, train_accuracy, valid_accuracy))
                    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            print('test accuracy %g' % accuracy.eval(feed_dict={x: ds.test.seqs, y_: ds.test.labels, keep_prob: 1.0}))
