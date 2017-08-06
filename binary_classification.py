import numpy as np
import time
import tensorflow as tf
from models import simple_network
from constants import *

class BinaryClassifier():
    def __init__ (self, hidden_size, data_sampler):
        self.data_sampler = data_sampler
        self.num_train = self.data_sampler.num_train

        self.hidden_size = hidden_size

        # inputs
        self.x = tf.placeholder(tf.float32, [None,] + self.data_sampler.x_dim , name='x')
        self.y = tf.placeholder(tf.int32, [None, self.data_sampler.n_classes], name='y')

        # computation graph
        self.net = simple_network(self.hidden_size)
        self.logits = self.net(self.x)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # Solver
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.solver = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss, var_list=self.net.vars)
        
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # logger file creation
        self.logfile = '/'.join(['logs', TASK_NAME])
        self.ckptfile = '/'.join(['checkpoints', TASK_NAME])
        if not tf.gfile.Exists(self.logfile):
            tf.gfile.MakeDirs(self.logfile)
        if not tf.gfile.Exists(self.ckptfile):
            tf.gfile.MakeDirs(self.ckptfile)

        with tf.name_scope('summaries'):
            loss_sum = tf.summary.scalar('loss', self.loss)
            acc_sum = tf.summary.scalr('accuracy', self.accuracy)
            self.summary_op = tf.summary.merge_all()

        # Saver
        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(self.logfile, self.sess.graph)

        # Checkpoints
        checkpoint = tf.train.get_checkpoint_state(self.ckptfile)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print ('[BC] Restore checkpoints from {}'.format(checkpoint.model_checkpoint_path))
        else:
            print ('[BC] Can not find checkpoint files')

    def train(self):
        ## init the weights
        self.sess.run(tf.global_variables_initializer())
        start_time = time.time()

        STEPS_PER_EPOCH = int (self.num_train/BATCH_SIZE)
        for t in range(0, NUM_EPOCH * STEPS_PER_EPOCH):
            batch_x, batch_y = self.data_sampler(BATCH_SIZE, is_train=True)
            _, cost = self.sess.run([self.solver, self.loss], feed_dict={self.y: batch_y, self.x: batch_x})

            if t % EVAL_PER_STEPS == 0:
                loss , summary = self.sess.run([self.loss, self.summary_op], feed_dict={self.y: batch_y, self.x: batch_x})
                self.writer.add_summary(summary, global_step=t)
                print('Iter [%8d] Time [%5.4f] Loss = %.4f ' % (t, time.time() - start_time, loss))

            # evaluation per epoch
            if t % STEPS_PER_EPOCH == 0:
                test_batch_x, test_batch_y = self.data_sampler(TEST_BATCH_SIZE, is_train=False)
                acc = self.sess.run([self.accuracy], feed_dict={self.x: test_batch_x, self.y: test_batch_y})
                print('Iter [%8d] Time [%5.4f] Accuracy = %.4f' % (t, time.time() - start_time, acc[0]))
