import tensorflow_datasets as tfds
from config import NUMBER_OF_EPOCHS
from discriminator import *
from gan_model import *
from generator import *
from monitor_callback import ModelMonitor
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

from save_callback import SaveWeights
from utils import *

print("STARTING MAIN")

# setup
setup_gpu()
delete_old_images()

# data pipeline
data_set = tfds.load('fashion_mnist', split='train')
data_set = data_set.map(normalize_image)
data_set = data_set.cache()
data_set = data_set.shuffle(60000)
data_set = data_set.batch(128)
data_set = data_set.prefetch(64)

print(data_set.as_numpy_iterator().next().shape)

generator_opt = Adam(learning_rate=0.0001)
generator_loss = BinaryCrossentropy()
discriminator_opt = Adam(learning_rate=0.00001)
discriminator_loss = BinaryCrossentropy()
generator = build_encoder()
discriminator = build_discriminator()

# Ładowanie checkpointu lub zaczynanie od zera jeśli folder jest pusty
if not os.listdir('load_checkpoint_from_here'):

    gan_model = GanModel(generator=generator,
                        discriminator=discriminator)

    gan_model.compile(generator_opt=generator_opt,
                  generator_loss=generator_loss,
                  discriminator_opt=discriminator_opt,
                  discriminator_loss=discriminator_loss)

    hist = gan_model.fit(data_set, epochs=NUMBER_OF_EPOCHS, callbacks=[ModelMonitor(), SaveWeights()])
else:
    generator.load_weights("load_checkpoint_from_here/generator.cpkt")
    discriminator.load_weights("load_checkpoint_from_here/discriminator.cpkt")

    gan_model = GanModel(generator=generator,
                         discriminator=discriminator)

    gan_model.compile(generator_opt=generator_opt,
                      generator_loss=generator_loss,
                      discriminator_opt=discriminator_opt,
                      discriminator_loss=discriminator_loss)

    hist = gan_model.fit(data_set, epochs=NUMBER_OF_EPOCHS, callbacks=[ModelMonitor(), SaveWeights()])

