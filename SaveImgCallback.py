import logging
import os
from random import random, randint
from zipfile import ZipFile

import gdown
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import save_img
from matplotlib import pyplot as plt
from functools import partial



class SaveImgCallback(tf.keras.callbacks.Callback):
    def __init__(self, num_img=9, z_dim=512):
        self.num_img = num_img
        self.z_dim = z_dim

    def on_train_batch_end(self, a, b):
        pass

    def on_epoch_end(self, epoch, logs):
        tf.random.set_seed(randint(1, 10000))
        z = tf.random.normal((self.num_img,1,1, self.z_dim))
        generated_images = self.model({"input": z, "alpha": 1.0})

        for i in range(self.num_img):
            res = "{:04d}".format(2 ** self.model.current_stage)
            epoch_str = "{:04d}".format(epoch)
            save_img("/data/fashion-gan/img/generated_img_res{res}_epoch{epoch}_{i}.png".format(res=res, i=i, epoch=epoch_str),
                     generated_images[i])