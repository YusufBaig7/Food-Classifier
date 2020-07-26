from preprocessing import preprocess
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd
from os import listdir
import PIL
import PIL.Image
from matplotlib.image import imread


train_images = []
train_labels = np.zeros(50)
apple_pie = preprocess('apple_pie')  # For the time being
samosa = preprocess('samosa')  # Messing around with only two classes

apple_pie = np.array(apple_pie)  # 4D numpy array
print(apple_pie.shape)
