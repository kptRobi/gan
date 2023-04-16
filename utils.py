import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, Dense, Reshape, LeakyReLU, UpSampling2D
from tensorflow.keras.models import Sequential


def setup_gpu():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print(tf.test.gpu_device_name())
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def normalize_image(data):
    image = data['image']
    return image / 255


def build_generator():
    model = Sequential()

    # input seed
    model.add(Dense(7 * 7 * 128, input_dim=128))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7, 7, 128)))

    # Upsampling block 1
    model.add(UpSampling2D())  # <- odwrócony pooling TODO (doczytać)
    model.add(Conv2D(128, 5, padding='same'))
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


def visualise_images(img):
    number_of_images = img.shape[0]
    fig, ax = plt.subplots(ncols=number_of_images, figsize=(20, 20))
    # Loop four times and get images
    for idx, img in enumerate(img):
        # Plot the image using a specific subplot
        ax[idx].imshow(np.squeeze(img))
        # Appending the image label as the plot title
        ax[idx].title.set_text(idx)
    plt.show()
