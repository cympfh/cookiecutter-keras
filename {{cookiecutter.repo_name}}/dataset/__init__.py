from keras.datasets import mnist
from keras.utils import np_utils


def load():
    """
    returns all training, validation data
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('f') / 255.0
    x_test = x_test.astype('f') / 255.0
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


def batch_generator():
    """
    yiled training, validation batches
    """
    yield None, None
