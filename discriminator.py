from tensorflow.keras.layers import Conv2D, Dense, Flatten, LeakyReLU, Dropout
from tensorflow.keras.models import Sequential


def build_discriminator():
    model = Sequential()

    # Convolutional block 1
    model.add(Conv2D(32, 5, input_shape=(28, 28, 1)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    # Convolutional block 2
    model.add(Conv2D(64, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    # Convolutional block 3
    model.add(Conv2D(128, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    # Convolutional block 4
    model.add(Conv2D(256, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    # Output
    model.add(Flatten())
    model.add(Dropout(0.4))  # TODO sprawdzić czy nie warto usunąć
    model.add(Dense(1, activation='sigmoid'))

    return model
