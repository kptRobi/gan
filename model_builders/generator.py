from keras import Input, Model, activations
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Reshape, LayerNormalization, UpSampling2D, Activation
from tensorflow import identity, reshape

from config import config
from main_package.utils_package.sparse_image_warp import sparse_image_warp


# TODO
# - pooling?
# -

def build_generator():
    image_input = Input(shape=(28, 28, 1), name="image_input")

    # ENKODER

    # blok 1
    e = Conv2D(64, 7, activation="relu")(image_input)
    e = Conv2D(128, 4, activation='relu')(e)
    e = Conv2D(256, 4, activation='relu')(e)

    # blok 2
    e = Conv2D(256, 3, activation='relu')(e)
    e = Conv2D(256, 3, bias_initializer=None)(e)
    e = Conv2D(256, 3, activation='relu')(e)
    e = Conv2D(256, 3, bias_initializer=None)(e)
    e = Conv2D(256, 3, activation='relu')(e)
    feature_map = Conv2D(256, 3, bias_initializer=None)(e)

    # KONTROLER STYLU

    # blok 1
    s = Conv2D(64, 7, activation="relu")(image_input)
    s = Conv2D(64, 7, activation='relu')(s)
    s = Conv2D(128, 4, activation='relu')(s)
    s = Conv2D(256, 4, activation='relu')(s)
    s = AveragePooling2D(2)(s)
    s = Flatten()(s)
    s = Dense(11)(s)

    # blok 2
    s = Dense(11)(s)
    s = Dense(11)(s)

    # gamma
    gamma = Dense(11)(s)
    gamma = Reshape((-1, 11))(gamma)
    gamma = "zeros"

    # beta
    beta = Dense(11)(s)
    beta = Reshape((-1, 11))(beta)
    beta = "zeros"

    # DEKODER

    # blok 1
    d = Conv2D(256, 3, activation='relu', padding='same')(feature_map)
    # d = LayerNormalization(gamma_initializer=gamma, beta_initializer=beta)(d)
    d = Conv2D(256, 3, padding='same')(d)
    # d = LayerNormalization(gamma_initializer=gamma, beta_initializer=beta)(d)
    d = Conv2D(256, 3, activation='relu', padding='same')(d)
    # d = LayerNormalization(gamma_initializer=gamma, beta_initializer=beta)(d)
    d = Conv2D(256, 3, padding='same')(d)
    # d = LayerNormalization(gamma_initializer=gamma, beta_initializer=beta)(d)
    d = Conv2D(256, 3, activation='relu', padding='same')(d)
    # d = LayerNormalization(gamma_initializer=gamma, beta_initializer=beta)(d)

    # blok 2
    d = UpSampling2D(2)(d)
    d = Conv2D(128, 5, activation='relu')(d)
    d = UpSampling2D(2)(d)
    d = Conv2D(64, 5, activation='relu', padding='same')(d)
    d = Conv2D(64, 5, padding='same')(d)
    decoded_image = Activation(activations.tanh)(d)

    # WARP KONTROLER

    # blok 1
    w = Flatten()(feature_map)
    w = Dense(11)(w)

    # punkty
    ldmark_pred = Dense(11)(w)

    # przemieszczenia
    ldmark_diff = Dense(11)(w)

    warp_input = identity(decoded_image, name='warp_input')
    src_pts = reshape(ldmark_pred, [-1, config.NUMBER_OF_LANDMARKS, 2])
    dst_pts = reshape(ldmark_pred + ldmark_diff, [-1, config.NUMBER_OF_LANDMARKS, 2])

    images_transformed, dense_flow = sparse_image_warp(warp_input, src_pts, dst_pts, regularization_weight = 1e-6, num_boundary_points=0)
    # images_transformed = None
    generator = Model(image_input, images_transformed, name="generator")
    return generator
