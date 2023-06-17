from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, LeakyReLU, Reshape, UpSampling2D, Conv2D

from config import config as config

# TODO
# - padding?
# -

def build_encoder():

    model = Sequential()

    # 1 BLOCK
    model.add(Conv2D( 64, 7, activation='relu', stride=1, input_shape=config.INPUT_SHAPE))
    model.add(Conv2D(128, 4, activation='relu', stride=2))
    model.add(Conv2D(256, 4, activation='relu', stride=2))

    # 2 BLOCK
    model.add(Conv2D(256, 3, activation='relu'))
    model.add(Conv2D(256, 3, biases_initializer=None))
    model.add(Conv2D(256, 3, activation='relu'))
    model.add(Conv2D(256, 3, biases_initializer=None))
    model.add(Conv2D(256, 3, activation='relu'))
    model.add(Conv2D(256, 3, biases_initializer=None))

    return model
