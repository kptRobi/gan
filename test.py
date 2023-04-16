import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt

from utils import *

test_generator = build_generator()
print(test_generator.summary())
img = test_generator.predict(np.random.randn(4, 128))
print(img.shape)
visualise_images(img)