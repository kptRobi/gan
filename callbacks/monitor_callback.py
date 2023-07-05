import os

import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import array_to_img
from datetime import date

from config import config


class ModelMonitor(Callback):
    def __init__(self, num_img=config.NUMBER_OF_IMAGES, latent_dim=config.LATENT_DIM):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.uniform((self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = array_to_img(generated_images[i])
            img.save(os.path.join('../images', f'generated_img_{epoch}_{i}.png'))

