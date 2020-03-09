from keras import optimizers
from keras.layers import Conv2D, Dense, Flatten, Reshape
from keras.models import Sequential

from {{cookiecutter.exec_name}}.config import Config


def build(config: Config):
    model = Sequential()
    model.add(Reshape((28, 28, 1), input_shape=(28, 28)))
    model.add(Conv2D(8, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(16, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    if config('opt') == 'SGD':
        opt = optimizers.SGD(clipvalue=1.0)
    elif config('opt') == 'Adam':
        opt = optimizers.Adam(clipvalue=1.0)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model
