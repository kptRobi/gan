import os

import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import array_to_img
from datetime import date
from config import LATENT_DIM, NUMBER_OF_IMAGES, FASHION_PATH


class ModelMonitor(Callback):
    def __init__(self, num_img=9, latent_dim=LATENT_DIM):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.uniform((self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = array_to_img(generated_images[i])
            epoch_str = "{:04d}".format(epoch)
            img.save(f'{FASHION_PATH}img/image_{epoch_str}_{i}.png')

