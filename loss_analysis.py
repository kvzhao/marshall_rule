from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import time
import sys
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

import matplotlib.pyplot as plt

from data_sampler import DataSampler
from rnn_model import RNN


print ('Version of tensorflow is {}'.format(tf.__version__))

# management
tf.app.flags.DEFINE_bool("is_train", False, "Set true for training, false flag will launch testing and analysis")
tf.app.flags.DEFINE_string("output_name", "results", "Assign objective of task")
tf.app.flags.DEFINE_string("model_name", "mytask_8x2x0.001", "Directly load the trained model by name")
tf.app.flags.DEFINE_string("DATA_PATH", "datasetMerged/states_j2j1.txt", "Set path of states file")
tf.app.flags.DEFINE_string("LABEL_PATH", "datasetMerged/sign_j2j1.txt", "Path to file of sign")
tf.app.flags.DEFINE_float("TRAINSET_RATIO", 0.0, "Assign ration of training set and reset for testing")

#TODO: Cellsize should refer to model_name
tf.app.flags.DEFINE_integer("cell_size", 8, "Size of lstm cells")
tf.app.flags.DEFINE_integer("num_output", 2, "Number of classes")

FLAGS = tf.app.flags.FLAGS

gpu_options = tf.GPUOptions(allow_growth=True)

def get_weights_by_name (sess, name):
    var = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith(name)]
    return np.squeeze(sess.run(var))

# add task_name parser

"""
    MAIN PROGRAM
"""

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    data_sampler = DataSampler(FLAGS.DATA_PATH, FLAGS.LABEL_PATH, FLAGS.TRAINSET_RATIO)
    num_train_data = data_sampler.num_train
    num_test_data  = data_sampler.num_test


    ckptfile = '/'.join(['checkpoints', FLAGS.model_name])
    if not tf.gfile.Exists(ckptfile):
        sys.exit ("Model {} not exist!".format(FLAGS.model_name))
        #tf.gfile.MakeDirs(ckptfile)
    if not tf.gfile.Exists(FLAGS.output_name):
        tf.gfile.MakeDirs(FLAGS.output_name)

    print ('=== LOSS SPECTRUM ANALYSIS ===')

    x = tf.placeholder(tf.float32, [None, data_sampler.x_dim] , name='x')
    y = tf.placeholder(tf.float32, [None, data_sampler.n_classes], name='y')

    # allocate empty network
    net = RNN(x, cell_size=FLAGS.cell_size, out_size=FLAGS.num_output)
    # connect operators
    logits, cell_states = net()
    print(logits.shape)
    print (y.shape)
    prob_op = tf.nn.softmax (logits)
    each_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    cross_entropy = -tf.reduce_sum(y * tf.log(prob_op), reduction_indices=[1])

    # initialization  (not necessary)
    sess.run(tf.global_variables_initializer())

    # Load checkpoints
    checkpoint = tf.train.get_checkpoint_state(ckptfile)
    saver = tf.train.Saver()
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        var_names =  [v for v in tf.global_variables()]
        print ('Restore checkpoints from {} with {} trained variables'.format(checkpoint.model_checkpoint_path, len(var_names)))
        for var in var_names:
            print (var)
    else:
        sys.exit ('Can not running test without trained model, please train first, use option --is_train=True')

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    confusion_matrix = tf.contrib.metrics.confusion_matrix(labels=tf.argmax(y, 1), predictions=tf.argmax(logits, 1)) 

    start_time = time.time()

    FULL_SIZE = data_sampler.num_test
    test_batch_x, test_batch_y = data_sampler(FULL_SIZE, is_train=False)
    entropies, probs, acc, confmat, losses, states, cpred = sess.run([cross_entropy, prob_op, accuracy, 
                                                                    confusion_matrix, each_loss, cell_states,
                                                                    tf.cast(correct_prediction, tf.float32)],
                                                                feed_dict={x: test_batch_x, y: test_batch_y}) 

    print('Prediction Time = {}, Validation Accuracy = {} %'.format((time.time() - start_time), acc * 100.0))
    print('Confusion Matrix : \n\t{} \n\t{}'.format(confmat[0], confmat[1]))

    ### ANALYSIS ###
    js = test_batch_x[:,0]

    np.save(FLAGS.output_name+ '/h_states', states[0])
    np.save(FLAGS.output_name+ '/c_states', states[1])
    np.save(FLAGS.output_name+ '/losses', losses)
    np.save(FLAGS.output_name+ '/probs', probs)
    np.save(FLAGS.output_name+ '/accuracy', cpred)
    np.save(FLAGS.output_name+ '/j2j1', js)
    np.save(FLAGS.output_name+ '/x', test_batch_x)
    np.save(FLAGS.output_name+ '/y', test_batch_y)

    #JL = zip(js, losses)
    #JL = zip(js, entropies)
    JL = zip(js, cpred)

    JL.sort(key = lambda x : x[0])
    jset = list(set(js))
    jset.sort()
    group_losses = [[jl[1] for jl in JL if jl[0] == j] for j in jset]
    
    means = []
    stddevs = []
    for i, j in enumerate(jset):
        means.append(np.mean(group_losses[i]))
        stddevs.append(np.std(group_losses[i]))

    # saving group losses
    np.save(FLAGS.output_name+ '/jest', jset)
    np.save(FLAGS.output_name+ '/means', means)
    np.save(FLAGS.output_name+ '/stddevs', stddevs)
    # plotting
    #plt.errorbar(jset, means, stddevs, linestyle='None', marker='^')
    plt.semilogx(jset, means, linewidth=2, marker='^')
    plt.xlabel('J2/J1')
    plt.ylabel('Accuracy')
    #plt.ylabel('xentropy loss')
    plt.title('Train on J = [0.1, 0.5]')
    plt.savefig(FLAGS.output_name + '/loss_spectrum.png')
    plt.show()

