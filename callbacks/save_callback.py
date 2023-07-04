from datetime import date

from tensorflow.keras.callbacks import Callback


class SaveWeights(Callback):
    def __init__(self):
        pass

    def on_epoch_end(self, epoch, logs=None):
        today = date.today().strftime("%m-%d")
        self.model.generator.save_weights(f"checkpoints/generator-{epoch}.cpkt")
        self.model.discriminator.save_weights(f"checkpoints/discriminator-{epoch}.cpkt")
