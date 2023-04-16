import tensorflow_datasets as tfds

from utils import setup_gpu, normalize_image

print("STARTING MAIN")

# setup
setup_gpu()

# data pipeline
data_set = tfds.load('fashion_mnist', split='train')
data_set = data_set.map(normalize_image)
data_set = data_set.cache()
data_set = data_set.shuffle(60000)
data_set = data_set.batch(128)
data_set = data_set.prefetch(64)

print(data_set.as_numpy_iterator().next().shape)
