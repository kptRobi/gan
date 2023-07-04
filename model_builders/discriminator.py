from keras.layers import Reshape
from tensorflow.keras.layers import Conv2D, Dense, Flatten, LeakyReLU, Dropout
from tensorflow.keras.models import Sequential


def build_discriminator():
    model = Sequential()

    model.add(Conv2D(32, 4, activation='relu'))
    model.add(Conv2D(64, 4, activation='relu'))
    model.add(Conv2D(128, 4, activation='relu'))
    model.add(Conv2D(256, 4, activation='relu'))
    model.add(Conv2D(512, 4, activation='relu'))

    return model

def build_patch_logits_model():
    model = Sequential()

    model.add(Conv2D(3, 1))
    model.add(Reshape())

    return model

def build_logits_model():
    model = Sequential()

    model.add(Flatten())
    model.add(Dense())
    model.add(Dense())

    return model
