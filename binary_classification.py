from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import time
import tensorflow as tf
from models import simple_network
from constants import *

class BinaryClassifier():
    def __init__ (self, data_sampler,
                    task_name,
                    hidden_sizes,
                    solver_type = 'adam', 
                    activation = 'relu',
                    loss_func = 'softmax_cross_entropy',
                    learning_rate = 0.001):
        self.data_sampler = data_sampler
        self.num_train = self.data_sampler.num_train

        self.task_name = task_name
        self.solver_type = solver_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.loss_func = loss_func
        self.learning_rate = learning_rate

        # inputs
        self.x = tf.placeholder(tf.float32, [None, self.data_sampler.x_dim] , name='x')
        self.y = tf.placeholder(tf.int32, [None, self.data_sampler.n_classes], name='y')

        # computation graph
        self.net = simple_network(self.hidden_sizes, activation=self.activation)
        # building network
        self.logits = self.net(self.x)

        self.regloss = 0.01 * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # (TODO) add options
        if self.loss_func == 'xentropy':
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        elif self.loss_func == 'hinge':
            self.loss = tf.reduce_mean(tf.losses.hinge_loss(logits=self.logits, labels=self.y))
        self.total_loss = self.loss + self.regloss

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        self.confusion_matrix = tf.contrib.metrics.confusion_matrix(labels=tf.argmax(self.y, 1), predictions=tf.argmax(self.logits, 1)) 

        # Solver
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            if self.solver_type == 'adam':
                self.solver = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss, var_list=self.net.vars)
            elif self.solver_type == 'sgd':
                self.solver = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss, var_list=self.net.vars)
        
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # logger file creation
        self.logfile = '/'.join(['logs', self.task_name])
        self.ckptfile = '/'.join(['checkpoints', self.task_name])
        if not tf.gfile.Exists(self.logfile):
            tf.gfile.MakeDirs(self.logfile)
        if not tf.gfile.Exists(self.ckptfile):
            tf.gfile.MakeDirs(self.ckptfile)

        with tf.name_scope('summaries'):
            tloss_sum = tf.summary.scalar('total loss', self.total_loss)
            loss_sum = tf.summary.scalar(self.loss_func+'_loss', self.loss)
            reg_sum = tf.summary.scalar('reg', self.regloss)
            acc_sum = tf.summary.scalar('accuracy', self.accuracy)
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
                loss = self.sess.run([self.loss], feed_dict={self.y: batch_y, self.x: batch_x})
                print('Iter [%8d] Time [%5.4f] Training Loss = %.4f ' % (t, time.time() - start_time, loss[0]))

            # evaluation per epoch
            if t % STEPS_PER_EPOCH == 0:
                test_batch_x, test_batch_y = self.data_sampler(TEST_BATCH_SIZE, is_train=False)
                loss, acc, summary, confmat = self.sess.run([self.loss, self.accuracy, self.summary_op, self.confusion_matrix],
                                        feed_dict={self.x: test_batch_x, self.y: test_batch_y})
                self.writer.add_summary(summary, global_step=t)
                print('Iter [%8d] Time [%5.4f] Validation Accuracy = %.4f' % (t, time.time() - start_time, acc))
                print('Confusion : \n\t{} \n\t{}'.format(confmat[0], confmat[1]))

    def overfit_test(self):
        """
            One of sanity-check techniques is overfitting test which exam the model 
            has capability to tackle one small piece of data.
        """
        self.sess.run(tf.global_variables_initializer())
        start_time = time.time()
        STEPS_PER_EPOCH = int (self.num_train/BATCH_SIZE)

        # only fetch data once
        batch_x, batch_y = self.data_sampler(BATCH_SIZE, is_train=True)
        for t in range(0, NUM_EPOCH * STEPS_PER_EPOCH):
            _, cost = self.sess.run([self.solver, self.loss], feed_dict={self.y: batch_y, self.x: batch_x})
            # show training loss
            if t % EVAL_PER_STEPS == 0:
                summary, loss = self.sess.run([self.summary_op, self.loss], feed_dict={self.y: batch_y, self.x: batch_x})
                print('Iter [%8d] Time [%5.4f] Training Loss = %.4f ' % (t, time.time() - start_time, loss))
                self.writer.add_summary(summary, global_step=t)

                if (loss <= 1e-3):
                    print ('Overfit test is done!')
                    break