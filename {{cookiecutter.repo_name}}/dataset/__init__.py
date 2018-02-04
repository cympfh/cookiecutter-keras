{%- if cookiecutter.fit_generator == "no" -%}
import numpy
from keras.datasets import mnist


def load(test=False):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if not test:
        X = x_train.astype('f') / 255.0
        y = y_train
    else:
        X = x_test.astype('f') / 255.0
        y = y_test
    return X, y
{%- else -%}
import math
import random

import numpy
from keras.datasets import mnist
from keras import utils


class Sequence(utils.Sequence):

    def __init__(self, X, y, batch_size, indices=None, test=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.indices = indices or list(range(len(X)))
        self.test = test

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, idx):
        test = self.test
        begin = idx * self.batch_size
        end = begin + self.batch_size
        batch_idx = self.indices[begin: end]
        batch_x = self.X[batch_idx]
        batch_y = self.y[batch_idx]

        # augmentation
        if not test:
            batch_x += numpy.random.normal(size=batch_x.shape, scale=0.01)

        return batch_x, batch_y


def batch_generator(batch_size, validation_split=0.1, test=False):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if not test:
        X = x_train.astype('f') / 255.0
        num = len(X)
        indices = list(range(num))
        random.shuffle(indices)
        num_valid = int(num * validation_split)
        indices_train = indices[num_valid:]
        indices_valid = indices[:num_valid]
        seq_train = Sequence(X, y_train, batch_size, indices=indices_train)
        seq_valid = Sequence(X, y_train, batch_size, indices=indices_valid)
        return seq_train, seq_valid
    else:
        X = x_test.astype('f') / 255.0
        return Sequence(X, y_test, batch_size, test=True)
{%- endif -%}
