import tensorflow as tf
import numpy as np
from enhancersdata import EnhancersData

data_dir_path = '/cs/grad/pazbu/paz/dev/projects/dnanet-v2/data'
ds = EnhancersData(data_dir_path)


with tf.Session() as sess:

    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('chk-0.936961433878-120900.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    x = tf.placeholder(tf.float32, shape=[None, 4, 1000])
    y_ = tf.placeholder(tf.uint8, shape=[None, 2])
    keep_prob = tf.placeholder(tf.float32)

    graph = tf.get_default_graph()
    y_pred_sig = graph.get_tensor_by_name("sigmoid_out:0")
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_pred_sig, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    all_outcomes = []
    batch_iter = ds.test.single_pass_batch_iter(100)
    num_batches = np.ceil(ds.test.num_examples / 100)
    c = 0
    for x_batch, y_batch in batch_iter:
        out = correct_prediction.eval(feed_dict={"x:0": x_batch, y_:y_batch, "keep_prob:0": 1.0})
        all_outcomes.extend(out)
        c += 1
        print("{:.2%}".format(c/num_batches))

    print('test accuracy:', sum(all_outcomes)/ds.test.num_examples)
