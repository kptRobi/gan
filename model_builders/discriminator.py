from keras import Input, Model
from keras.layers import Conv2D, Flatten, Dense, Reshape


# TODO
# - pooling?
# -

def build_generator():
    image_input = Input(shape=(28, 28, 1), name="image_input")

    # blok 1
    d = Conv2D(32, 4, activation='relu')(image_input)
    d = Conv2D(64, 4, activation='relu')(d)
    d = Conv2D(128, 4, activation='relu')(d)
    d = Conv2D(256, 4, activation='relu')(d)
    d = Conv2D(512, 4, activation='relu')(d)

    patch_logits = Conv2D(3, 1)(d)
    patch_logits = Reshape()(patch_logits)

    logits = Flatten()(d)
    logits = Dense()(logits)
    logits = Dense()(logits)

    discriminator = Model(inputs=[image_input],
                          outputs=[patch_logits, logits],
                          name="discriminator")
    return discriminator
