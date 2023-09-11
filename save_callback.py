from datetime import date

from tensorflow.keras.callbacks import Callback

from config import FASHION_PATH


class SaveWeights(Callback):
    def __init__(self):
        pass

    def on_epoch_end(self, epoch, logs=None):
        today = date.today().strftime("%m-%d")
        self.model.generator.save_weights(f"{FASHION_PATH}checkpoints/generator-{epoch}.cpkt")
        self.model.discriminator.save_weights(f"{FASHION_PATH}checkpoints/discriminator-{epoch}.cpkt")
