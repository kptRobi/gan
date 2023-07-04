from keras import layers, activations
from keras.layers import LayerNormalization
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, LeakyReLU, Reshape, UpSampling2D, Conv2D
from tensorflow.python.util._pywrap_utils import Flatten

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

def build_decoder():
    model = Sequential()

    # 1 BLOCK
    model.add(Conv2D(256, 3, activation='relu'))
    model.add(LayerNormalization())
    model.add(Conv2D(256, 3))
    model.add(LayerNormalization())
    model.add(Conv2D(256, 3, activation='relu'))
    model.add(LayerNormalization())
    model.add(Conv2D(256, 3))
    model.add(LayerNormalization())
    model.add(Conv2D(256, 3, activation='relu'))
    model.add(LayerNormalization())

    model.add(UpSampling2D(2))
    model.add(Conv2D(128, 5, activation='relu'))
    model.add(UpSampling2D(2))
    model.add(Conv2D(64, 5, activation='relu'))
    model.add(Conv2D(64, 5))
    model.add(layers.Activation(activations.tanh))

    return model

def build_warp_controller():
    model = Sequential()

    model.add(Flatten())
    model.add(Dense())

    return model

def build_control_points_model():
    model = Sequential()

    model.add(Dense())

    return model

def build_warp_points_model():
    model = Sequential()

    model.add(Dense())

    return model
