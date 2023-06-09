from config import NUMBER_OF_IMAGES
from development import *

# TEST GENERATORA
test_generator = build_encoder()
print(test_generator.summary())
img = test_generator.predict(np.random.randn(NUMBER_OF_IMAGES, LATENT_DIM))
visualise_images(img)

# TESTY DYSKRYMINATORA
test_discriminator = build_discriminator()
print(test_discriminator.summary())
print(test_discriminator.predict(img))