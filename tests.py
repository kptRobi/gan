from generator import *
from discriminator import *
from utils import *
from development import *

# CONFIG:
number_of_images = 3

# TEST GENERATORA
test_generator = build_generator()
print(test_generator.summary())
img = test_generator.predict(np.random.randn(number_of_images, 128))
visualise_images(img)

# TESTY DYSKRYMINATORA
test_discriminator = build_discriminator()
print(test_discriminator.summary())
print(test_discriminator.predict(img))