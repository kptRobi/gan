import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt

from utils import setup_gpu

print("Hello woraaaald")

#setup
setup_gpu()
data_set = tfds.load('fashion_mnist', split='train')
iterator = data_set.as_numpy_iterator()
# Setup the subplot formatting
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
# Loop four times and get images
for idx in range(4):
    # Grab an image and label
    sample = iterator.next()
    # Plot the image using a specific subplot
    ax[idx].imshow(np.squeeze(sample['image']))
    # Appending the image label as the plot title
    ax[idx].title.set_text(sample['label'])
plt.show()