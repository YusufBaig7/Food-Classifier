import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd
from os import listdir
import PIL
import PIL.Image
from matplotlib.image import imread


def preprocess(imagename):
    train_images = []
    #train_labels = []
    #check = np.zeros((512, 512, 3))
    f = listdir('images/' + imagename)
    for i in range(50):  # Just going through first 50 elements for now
        image = imread('images/' + imagename + '/' + f[i])
        a1, a2, a3 = image.shape  # Storing image dimensions
        if (a1 == 512 & a2 == 512):  # Checking for 512x512 images
            train_images.append(image)

    return (train_images)  # returning list of pixel matrices
