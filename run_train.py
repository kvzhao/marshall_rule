import os
import tensorflow as tf
from data_sampler import DataSampler
from binary_classification import BinaryClassifier

tf.flags.DEFINE_boolean("sanity_check", False, "If set, training will perform on small piece of data.")
FLAGS = tf.flags.FLAGS

def main():
    for loss_type in ['xentropy','hinge']:
        for h1 in range(5):
            hidden_sizes = [2 ** (h1+1)]
            ds = DataSampler()
            arch = 'x'.join([str(i) for i in hidden_sizes])
            task = '_'.join([arch, loss_type])
            print ('[TRAIN] Start experiment: {}'.format(task))
            classifier = BinaryClassifier(data_sampler=ds, 
                                        task_name = task,
                                        hidden_sizes=hidden_sizes,
                                        loss_func=loss_type)
            if (FLAGS.sanity_check):
                classifier.overfit_test()
            else:
                classifier.train()
            # reset the model
            tf.reset_default_graph()
            print ('[TRAIN] Done {}, reset network.'.format(task))

if __name__ == '__main__':
    main()