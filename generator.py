from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, LeakyReLU, Reshape, UpSampling2D, Conv2D

from config import LATENT_DIM


def build_generator():
    model = Sequential()

    # input seed
    model.add(Dense(7*7*128, input_dim=128))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7, 7, 128)))

    # Upsampling block 1
    model.add(UpSampling2D())  # <- odwrócony pooling TODO (doczytać)
    model.add(Conv2D(128, 5, padding='same'))   # <- nie zmniejsza rozmiaru
    model.add(LeakyReLU(0.2))

    # Upsampling block 2
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))

    # Convolutional block 1
    model.add(Conv2D(128, 4, padding='same'))
    model.add(LeakyReLU(0.2))

    # Convolutional block 2
    model.add(Conv2D(128, 4, padding='same'))
    model.add(LeakyReLU(0.2))

    # Convolutional output layer
    model.add(Conv2D(1, 4, padding='same', activation='sigmoid'))

    return model
