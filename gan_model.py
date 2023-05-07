import tensorflow as tf
from tensorflow.keras.models import Model

from config import NOISE, LATENT_DIM


class GanModel(Model):
    def __init__(self, generator, discriminator, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.generator = generator
        self.discriminator = discriminator

    def compile(self, generator_opt, generator_loss, discriminator_opt, discriminator_loss, *args, **kwargs):
        super().compile(*args, **kwargs)

        self.generator_opt = generator_opt
        self.generator_loss = generator_loss
        self.discriminator_opt = discriminator_opt
        self.discriminator_loss = discriminator_loss

    def train_step(self, batch):
        real_images = batch
        fake_images = self.generator(tf.random.normal((128, 128)), training=False)

        # TRAINING THE DISCRIMINATOR
        with tf.GradientTape() as discrriminator_tape:
            real_images_predictions = self.discriminator(real_images, training=True)
            fake_images_predictions = self.discriminator(fake_images, training=True)
            concatenated_predictions = tf.concat([real_images_predictions, fake_images_predictions], axis=0)

            concatenated_labels = tf.concat(
                [tf.zeros_like(real_images_predictions), tf.ones_like(fake_images_predictions)],
                axis=0)

            # noise TODO wypróbować różne wartości
            noise_real = 0.15*tf.random.uniform(tf.shape(real_images_predictions))
            noise_fake = -0.15*tf.random.uniform(tf.shape(fake_images_predictions))
            concatenated_predictions += tf.concat([noise_real, noise_fake], axis=0)

            total_discriminator_loss = self.discriminator_loss(concatenated_labels, concatenated_predictions)

        # liczy gradient TODO doczytać
        discriminator_gradient = discrriminator_tape.gradient(total_discriminator_loss,
                                                              self.discriminator.trainable_variables)
        # TODO czy to jest backpropagation?
        self.discriminator_opt.apply_gradients(zip(discriminator_gradient, self.discriminator.trainable_variables))

        # TRAINING THE GENERATOR
        with tf.GradientTape() as generator_tape:
            generated_images = self.generator(tf.random.normal((128,128)), training=True)

            predicted_labels = self.discriminator(generated_images, training=False)

            total_generator_loss = self.generator_loss(tf.zeros_like(predicted_labels), predicted_labels)

        generator_gradient = generator_tape.gradient(total_generator_loss, self.generator.trainable_variables)
        # TODO czy to jest backpropagation?
        self.generator_opt.apply_gradients(zip(generator_gradient, self.generator.trainable_variables))

        return {"d_loss": total_discriminator_loss, "g_loss": total_generator_loss}
