import tensorflow_datasets as tfds
from config import NUMBER_OF_EPOCHS, FASHION_PATH
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

# data pipeline
data_set = tfds.load('fashion_mnist')
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
discriminator = build_discriminator()

gan_model = GanModel(generator=generator,
                     discriminator=discriminator)

gan_model.compile(generator_opt=generator_opt,
                  generator_loss=generator_loss,
                  discriminator_opt=discriminator_opt,
                  discriminator_loss=discriminator_loss)

hist = gan_model.fit(data_set, epochs=1, callbacks=[ModelMonitor(), SaveWeights()])

np.save(f'{FASHION_PATH}history/history.npy', hist.history)
