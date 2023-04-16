import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt



data_set = tfds.load('fashion_mnist', split='train')
iterator = data_set.as_numpy_iterator()
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx in range(4):
    sample = iterator.next()
    ax[idx].imshow(np.squeeze(sample['image']))
    ax[idx].title.set_text(sample['label'])
plt.show()