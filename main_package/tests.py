import tensorflow_datasets as tfds
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

from main_package.gan_model import GanModel
from main_package.utils_package.utils import normalize_image
from model_builders.generator import build_generator

print("STARTING MAIN")

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
generator = build_generator()
discriminator = None

gan_model = GanModel(generator=generator,
                     discriminator=discriminator)

gan_model.compile(generator_opt=generator_opt,
                  generator_loss=generator_loss,
                  discriminator_opt=discriminator_opt,
                  discriminator_loss=discriminator_loss)
