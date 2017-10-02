from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import time
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from data_sampler import DataSampler
from rnn_model import RNN


print ('Version of tensorflow is {}'.format(tf.__version__))

# management
tf.app.flags.DEFINE_bool("is_train", True, "Set true for training, false flag will launch testing and analysis")
tf.app.flags.DEFINE_string("task_name", "mytask", "Assign objective of task")
tf.app.flags.DEFINE_string("DATA_PATH", "datasetMerged/states_J0.txt", "Set path of states file")
tf.app.flags.DEFINE_string("LABEL_PATH", "datasetMerged/sign_J0.txt", "Path to file of sign")
tf.app.flags.DEFINE_float("TRAINSET_RATIO", 0.6, "Assign ration of training set and reset for testing")
tf.app.flags.DEFINE_bool("single_time_step", False, "Set Ture when the sequence is feeded element by element")

#TODO: Add reset option

# hyper-params
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_integer("cell_size", 16, "Size of lstm cells")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of LSTM Layers")
tf.app.flags.DEFINE_integer("num_classes", 2, "Number of classes")
tf.app.flags.DEFINE_bool("use_cos", True, "Option to use Cos for first layer activation function")

# traning process
tf.app.flags.DEFINE_integer("NUM_EPOCH", 200, "Number of epochs")
tf.app.flags.DEFINE_integer("BATCH_SIZE", 32, "Batch size")
tf.app.flags.DEFINE_integer("TEST_BATCH_SIZE", 512, "Test Batch size")
tf.app.flags.DEFINE_integer("EVAL_PER_STEPS", 2000, "Steps between evaluation")
tf.app.flags.DEFINE_integer("SAVE_CKPT_PER_STEPS", 100000, "Steps interval for saving checkpoints")

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
    data_sampler = DataSampler(FLAGS.DATA_PATH, FLAGS.LABEL_PATH, FLAGS.TRAINSET_RATIO, SINGLE_TIME=FLAGS.single_time_step)
    num_train_data = data_sampler.num_train
    num_test_data  = data_sampler.num_test

    task_name = FLAGS.task_name + '_' + 'x'.join([str(FLAGS.num_layers),
                                                str(FLAGS.cell_size),
                                                str(FLAGS.num_classes),
                                                str(FLAGS.learning_rate)])

    logfile = '/'.join(['logs', task_name])
    ckptfile = '/'.join(['checkpoints', task_name])
    outfile = '/'.join(['lstmw_'+task_name])

    if not tf.gfile.Exists(logfile):
        tf.gfile.MakeDirs(logfile)
    if not tf.gfile.Exists(ckptfile):
        tf.gfile.MakeDirs(ckptfile)
    if not FLAGS.is_train and not tf.gfile.Exists(outfile):
        tf.gfile.MakeDirs(outfile)

    if FLAGS.is_train:
        print ('Training ...')

        # Batch size x time steps x features.
        #if FLAGS.single_time_step:
        #    x = tf.placeholder(tf.float32, [None, data_sampler.x_dim, 1] , name='x')
        #else:
        x = tf.placeholder(tf.float32, [None, 1, data_sampler.x_dim] , name='x')
        y = tf.placeholder(tf.int32, [None, data_sampler.n_classes], name='y')
        x_len = tf.placeholder(tf.int32, [None, ], name='x_len')

        net = RNN(x=x, x_len=x_len,
                    cell_size=FLAGS.cell_size,
                    num_classes=FLAGS.num_classes,
                    num_layers=FLAGS.num_layers,
                    use_cos=FLAGS.use_cos
                    )

        logits, _ = net()
        prob_op = tf.nn.softmax (logits)

        # loss function
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        correct_prediction = tf.equal(tf.argmax(prob_op, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        confusion_matrix = tf.contrib.metrics.confusion_matrix(labels=tf.argmax(y, 1), predictions=tf.argmax(logits, 1))

        # Solver
        solver = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss, var_list=net.vars)

        # Saver
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(logfile, sess.graph)

        with tf.name_scope('summaries'):
            loss_sum = tf.summary.scalar('loss', loss)
            acc_sum = tf.summary.scalar('accuracy', accuracy)
            summary_op = tf.summary.merge_all()

        # Checkpoints
        checkpoint = tf.train.get_checkpoint_state(ckptfile)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print ('Restore checkpoints from {}'.format(checkpoint.model_checkpoint_path))
        else:
            print ('Can not find checkpoint files, run new experiment')

        # training loop
        start_time = time.time()
        stable_counter = 0

        STEPS_PER_EPOCH = int (num_train_data/FLAGS.BATCH_SIZE)
        print ('{} steps per epoch'.format(STEPS_PER_EPOCH))

        # initialize the variables before running expierment
        sess.run(tf.global_variables_initializer())

        for t in range(0, FLAGS.NUM_EPOCH * STEPS_PER_EPOCH):
            batch_x, batch_y = data_sampler(FLAGS.BATCH_SIZE, is_train=True)
            # Feed length here
            batch_xlen = np.array([data_sampler.x_dim] * FLAGS.BATCH_SIZE, dtype=np.int32)
            _, cost = sess.run([solver, loss], feed_dict={y: batch_y, x: batch_x, x_len: batch_xlen})
            if t % FLAGS.EVAL_PER_STEPS == 0:
                losses = sess.run([loss], feed_dict={y: batch_y, x: batch_x, x_len: batch_xlen})
                print('Iter [%8d] Time [%5.4f] Training Loss = %.4f ' % (t, time.time() - start_time, losses[0]))
            if t % FLAGS.SAVE_CKPT_PER_STEPS == 0:
                print ('Save to checkpoint')
                saver.save(sess, ckptfile + '/model', global_step=t)
                # evaluation per epoch
            if t % STEPS_PER_EPOCH == 0:
                test_batch_x, test_batch_y = data_sampler(FLAGS.TEST_BATCH_SIZE, is_train=False)
                test_batch_xlen = np.array([data_sampler.x_dim] * FLAGS.TEST_BATCH_SIZE, dtype=np.int32)
                losses, acc, summary, confmat = sess.run([loss, accuracy, summary_op, confusion_matrix],
                                                            feed_dict={x: test_batch_x, y: test_batch_y, x_len: test_batch_xlen})
                writer.add_summary(summary, global_step=t)
                print('Iter [%8d] Time [%5.4f] Validation Accuracy = %.4f' % (t, time.time() - start_time, acc))
                print('Confusion : \n\t{} \n\t{}'.format(confmat[0], confmat[1]))
                if (acc >= 0.97):
                    stable_counter += 1
                    if (stable_counter >= 1000):
                        print ('Validation Accuracy exceeds 97%, the task terminates.')
                        break
    else: # is_train=false
        print ('Testing')

        #if FLAGS.single_time_step:
        #    x = tf.placeholder(tf.float32, [None, data_sampler.x_dim, 1] , name='x')
        #else:
        x = tf.placeholder(tf.float32, [None, 1, data_sampler.x_dim] , name='x')
        y = tf.placeholder(tf.int32, [None, data_sampler.n_classes], name='y')

        # allocate empty network
        net = RNN(x, cell_size=FLAGS.cell_size,
                    num_classes=FLAGS.num_classes,
                    num_layers=FLAGS.num_layers,
                    use_cos=FLAGS.use_cos
                    )
        # connect operators
        logits, cell_states = net()
        prob_op = tf.nn.softmax (logits)

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

        correct_prediction = tf.equal(tf.argmax(prob_op, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        confusion_matrix = tf.contrib.metrics.confusion_matrix(labels=tf.argmax(y, 1), predictions=tf.argmax(logits, 1))

        start_time = time.time()

        test_batch_x, test_batch_y = data_sampler(FLAGS.TEST_BATCH_SIZE, is_train=False)
        probs, acc, confmat, lstm_states = sess.run([prob_op, accuracy, confusion_matrix, cell_states], feed_dict={x: test_batch_x, y: test_batch_y})

        print('Prediction Time = {}, Testing Accuracy = {} %'.format((time.time() - start_time), acc * 100.0))
        print('Confusion Matrix : \n\t{} \n\t{}'.format(confmat[0], confmat[1]))

        """
            THE FOLLOWING CODES ARE USEFUL FOR FURTHER ANALYSIS
        """

        # shape [batch_size, cell_size]
        lstm_h = lstm_states[0]
        lstm_c = lstm_states[1]

        ## retrieve weights from loaded variables
        lstm_mat = get_weights_by_name(sess, 'kernel:0')
        lstm_bias =get_weights_by_name(sess, 'bias:0')
        linout_w = get_weights_by_name(sess, 'w:0')
        linout_b = get_weights_by_name(sess, 'b:0')

        """
            Meaning of subscript
            i: information gate
            c: memory gate (tranform layer)
            f: forget gate
            o: output gate
        """
        Wi, Wc, Wf, Wo = np.hsplit(lstm_mat, 4)
        bi, bc, bf, bo = np.split(lstm_bias, 4)

        print (Wi.shape)
        print (Wc.shape)
        print (Wf.shape)
        print (Wo.shape)
        print (bi.shape)
        print (bc.shape)
        print (bf.shape)
        print (bo.shape)
        print (linout_w.shape)
        print (linout_b.shape)

        np.save(outfile + '/Wi', Wi)
        np.save(outfile + '/Wc', Wc)
        np.save(outfile + '/Wf', Wf)
        np.save(outfile + '/Wo', Wo)
        np.save(outfile + '/bi', bi)
        np.save(outfile + '/bc', bc)
        np.save(outfile + '/bf', bf)
        np.save(outfile + '/bo', bo)
        np.save(outfile + '/linout_w', linout_w)
        np.save(outfile + '/linout_b', linout_b)
