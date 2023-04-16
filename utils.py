import tensorflow as tf
def setup_gpu():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print(tf.test.gpu_device_name())
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def normalize_image(data):
    image = data['image']
    return image / 255