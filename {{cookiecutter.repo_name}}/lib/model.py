from keras.layers import Conv2D, Dense, Flatten, Reshape
from keras.models import Sequential
from keras.optimizers import Adam


def build():
    model = Sequential()
    model.add(Reshape((28, 28, 1), input_shape=(28, 28)))
    model.add(Conv2D(8, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(16, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    opt = Adam(clipvalue=1.0)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model
