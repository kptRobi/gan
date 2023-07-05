import os
import shutil

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def setup_gpu():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print(tf.test.gpu_device_name())
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)


def normalize_image(data):
    image = data['image']
    return image / 255


def visualise_images(img):
    number_of_images = img.shape[0]
    fig, ax = plt.subplots(ncols=number_of_images, figsize=(20, 20))
    # Loop four times and get images
    for idx, img in enumerate(img):
        # Plot the image using a specific subplot
        ax[idx].imshow(np.squeeze(img))
        # Appending the image label as the plot title
        ax[idx].title.set_text(idx)
    plt.show()

def delete_old_images():
    folder_path = "../../images"
    if os.path.exists(folder_path):  # sprawdzenie, czy folder istnieje
        for file_name in os.listdir(folder_path):  # pętla po plikach w folderze
            file_path = os.path.join(folder_path, file_name)  # pełna ścieżka do pliku
            if os.path.isfile(file_path):  # sprawdzenie, czy plik jest plikiem
                os.remove(file_path)  # usunięcie pliku
        print("Zawartość folderu usunięta")
    else:
        print("Folder nie istnieje")