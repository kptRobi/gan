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

generator_opt = Adam(learning_rate=0.0001)
generator_loss = BinaryCrossentropy()
discriminator_opt = Adam(learning_rate=0.00001)
discriminator_loss = BinaryCrossentropy()
generator = build_generator()
discriminator = build_discriminator()

gan_model = GanModel(generator=generator,
                     discriminator=discriminator)

gan_model.compile(generator_opt=generator_opt,
                  generator_loss=generator_loss,
                  discriminator_opt=discriminator_opt,
                  discriminator_loss=discriminator_loss)
