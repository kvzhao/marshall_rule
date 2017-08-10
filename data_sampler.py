import numpy as np
import os, sys
import collections

from random import shuffle

def read_data_sets(data_path, label_path, normalized=True):
    ## read data, hard-coded
    images = np.loadtxt(data_path)
    labels = np.loadtxt(label_path)
    num_data = images.shape[0]
    ## 80% for training and 20% data for testing
    train_num = int(.8 * num_data)
    # shuffle and seperate training and testing set
    perm = np.arange(num_data)
    np.random.shuffle(perm)
    train_images = images[perm[:train_num]]
    train_labels = labels[perm[:train_num]]
    test_images = images[perm[train_num:]]
    test_labels = labels[perm[train_num:]]
    print ('Training set contains {} and testing set {}'.format(train_images.shape[0], test_images.shape[0]))

    trainset = DataSet(images=train_images, labels=train_labels, normalized=normalized)
    testset = DataSet(images=test_images, labels=test_labels, normalized=normalized)
    return trainset, testset

class DataSampler(object):
    def __init__ (self, DATA_PATH, LABEL_PATH, DATA_NORMALIZED=False):
        #TODO: dont use hard-coded
        self.train_set, self.test_set = read_data_sets(DATA_PATH, LABEL_PATH, normalized=DATA_NORMALIZED)
        self.num_train = self.train_set._num_of_samples
        self.num_test  = self.test_set._num_of_samples
        self.x_dim = self.train_set._image_shape[0]
        self.n_classes = self.train_set._label_shape[0]
        print ('Datasampler x_dim {} and n_classes {}'.format(self.x_dim, self.n_classes))

    def __call__(self, batch_size, is_train=True):
        if is_train:
            return self.train_set.next_batch(batch_size)
        else:
            return self.test_set.next_batch(batch_size)
    
class DataSet(object):
    def __init__(self, images, labels, dtpye=np.float32, normalized=True):
        self._images = images
        self._labels = labels
        self._num_of_samples = images.shape[0]
        self._image_dim = len(images.shape)
        self._image_shape = images.shape[1:]
        self._label_shape = labels.shape[1:]
        self._epochs_completed = 0
        self._index_in_epoch = 0
        print ('Image is {}D with shape={}. Label with shape={}'.format(self._image_dim-1, self._image_shape, self._label_shape))

        if normalized:
            self._normalized_data()

    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels
    @property
    def num_samples(self):
        return self._num_of_samples
    @property
    def epochs_completed(self):
        return self._epochs_completed

    def _normalized_data(self):
        print ('[DS] original dataset with mean = {}, and std = {}'.format(np.mean(self._images), np.std(self._images)))
        self._images[self._images == 0] = -1
        mean = np.mean(self._images)
        stddev = np.std(self._images)
        print ('[DS] remove zero, mean = {}, and std = {}'.format(mean, stddev))
        # convert 0 to -1 
        self._images = (self._images - mean) / stddev
        print ('[DS] processed dataset with mean = {}, and std = {}'.format(np.mean(self._images), np.std(self._images)))
        #  And shuffle the data
        perm = np.arange(self._num_of_samples)
        np.random.shuffle(perm)
        self._labels = self.labels[perm]

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        # Shuffle the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_of_samples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
                # Finsh the epoch
        if start + batch_size > self._num_of_samples:
            self._epochs_completed += 1
            rest_num_samples = self._num_of_samples - start
            images_rest_part = self._images[start:self._num_of_samples]
            labels_rest_part = self._labels[start:self._num_of_samples]
            # Shuffle
            if shuffle:
                perm = np.arange(self._num_of_samples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
                # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_samples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), \
                np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]
