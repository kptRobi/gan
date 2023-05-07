import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D

from config import NUMBER_OF_EPOCHS
from gan_model import *
from generator import *
from discriminator import *
from utils import *

# Adam is going to be the optimizer for both
from tensorflow.keras.optimizers import Adam
# Binary cross entropy is going to be the loss for both
from tensorflow.keras.losses import BinaryCrossentropy

g_opt = Adam(learning_rate=0.0001)
d_opt = Adam(learning_rate=0.00001)
g_loss = BinaryCrossentropy()
d_loss = BinaryCrossentropy()
generator = build_generator()
discriminator = build_discriminator()

gan_model = GanModel(generator, discriminator)

gan_model.compile(g_opt, d_opt, g_loss, d_loss)
