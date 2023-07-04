from keras import layers, activations
from keras.layers import LayerNormalization, AveragePooling2D
from tensorflow import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, LeakyReLU, Reshape, UpSampling2D, Conv2D
from tensorflow.python.util._pywrap_utils import Flatten
import tensorflow as tf

from config import config as config
from main.utils.sparse_image_warp import sparse_image_warp


# TODO
# - pooling?
# -

def build_generator():
    image_input = keras.Input(shape=(28, 28, 1), name="image_input")


    # ENKODER

        # blok 1
    e = Conv2D(64, 7, activation="relu")(image_input)
    e = Conv2D(128, 4, activation='relu', stride=2)(e)
    e = Conv2D(256, 4, activation='relu', stride=2)(e)

        # blok 2
    e = Conv2D(256, 3, activation='relu')(e)
    e = Conv2D(256, 3, biases_initializer=None)(e)
    e = Conv2D(256, 3, activation='relu')(e)
    e = Conv2D(256, 3, biases_initializer=None)(e)
    e = Conv2D(256, 3, activation='relu')(e)
    feature_map = Conv2D(256, 3, biases_initializer=None)(e)


    # KONTROLER STYLU

        # blok 1
    s = Conv2D(64, 7, activation="relu")(image_input)
    s = Conv2D(64, 7, activation='relu')(s)
    s = Conv2D(128, 4, activation='relu')(s)
    s = Conv2D(256, 4, activation='relu')(s)
    s = AveragePooling2D(2)(s)
    s = Flatten()(s)
    s = Dense()(s)

        # blok 2
    s = Dense()(s)
    s = Dense()(s)

        # gamma
    gamma = Dense()(s)
    gamma = Reshape()(gamma)

        # beta
    beta = Dense()(s)
    beta = Reshape()(beta)


    # DEKODER

        # blok 1
    d = Conv2D(256, 3, activation='relu')(feature_map)
    d = LayerNormalization(gamma_initializer=gamma, beta_initializer=beta)(d)
    d = Conv2D(256, 3)(d)
    d = LayerNormalization(gamma_initializer=gamma, beta_initializer=beta)(d)
    d = Conv2D(256, 3, activation='relu')(d)
    d = LayerNormalization(gamma_initializer=gamma, beta_initializer=beta)(d)
    d = Conv2D(256, 3)(d)
    d = LayerNormalization(gamma_initializer=gamma, beta_initializer=beta)(d)
    d = Conv2D(256, 3, activation='relu')(d)
    d = LayerNormalization(gamma_initializer=gamma, beta_initializer=beta)(d)

        # blok 2
    d = UpSampling2D(2)(d)
    d = Conv2D(128, 5, activation='relu')(d)
    d = UpSampling2D(2)(d)
    d = Conv2D(64, 5, activation='relu')(d)
    d = Conv2D(64, 5)(d)
    decoded_image = layers.Activation(activations.tanh)(d)


    # WARP KONTROLER

        # blok 1
    w = Flatten()(feature_map)
    w = Dense()(w)

        # punkty
    ldmark_pred = Dense()(w)

        # przemieszczenia
    ldmark_diff = Dense()(w)

    warp_input = tf.identity(decoded_image, name='warp_input')
    src_pts = tf.reshape(ldmark_pred, [-1, config.NUMBER_OF_LANDMARKS, 2])
    dst_pts = tf.reshape(ldmark_pred + ldmark_diff, [-1, config.NUMBER_OF_LANDMARKS, 2])

    images_transformed, dense_flow = sparse_image_warp(warp_input, src_pts, dst_pts, regularization_weight = 1e-6, num_boundary_points=0)

    generator = keras.Model(image_input, images_transformed,  name="generator")
    return generator
