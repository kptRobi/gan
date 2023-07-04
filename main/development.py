from gan_model import *
from model_builders.generator import *
from model_builders.discriminator import *

# Adam is going to be the optimizer for both
from tensorflow.keras.optimizers import Adam
# Binary cross entropy is going to be the loss for both
from tensorflow.keras.losses import BinaryCrossentropy

generator_opt = Adam(learning_rate=0.0001)
generator_loss = BinaryCrossentropy()
discriminator_opt = Adam(learning_rate=0.00001)
discriminator_loss = BinaryCrossentropy()
generator = build_encoder()
discriminator = build_discriminator()

gan_model = GanModel(generator=generator,
                     discriminator=discriminator)

gan_model.compile(generator_opt=generator_opt,
                  generator_loss=generator_loss,
                  discriminator_opt=discriminator_opt,
                  discriminator_loss=discriminator_loss)
