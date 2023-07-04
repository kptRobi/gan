from keras import layers, activations
from keras.layers import LayerNormalization, AveragePooling2D, Flatten
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, LeakyReLU, Reshape, UpSampling2D, Conv2D

from config import config as config


def build_image_extraction_model():
    model = Sequential()

    # 1 BLOCK
    model.add(Conv2D(64, 7, activation='relu'))
    model.add(Conv2D(128, 4, activation='relu'))
    model.add(Conv2D(256, 4, activation='relu'))

    model.add(AveragePooling2D(2))
    model.add(Flatten())
    model.add(Dense())

    return model

def build_middle_model():
    model = Sequential()

    model.add(Dense())
    model.add(Dense())

    return model

def build_gamma_model():
    model = Sequential()

    model.add(Dense())
    model.add(Reshape)

    return model


def build_beta_model():
    model = Sequential()

    model.add(Dense())
    model.add(Reshape)

    return model
