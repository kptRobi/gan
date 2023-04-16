from utils import *

# CONFIG:
number_of_images = 3

# TEST GENERATORA
test_generator = build_generator()
print(test_generator.summary())
img = test_generator.predict(np.random.randn(number_of_images, 128))
visualise_images(img)