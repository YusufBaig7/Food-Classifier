import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import tensorflow as tf

from tensorflow.keras import layers

from tensorflow.keras import Model

import os

from os import getcwd

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

	path_vg = f"{getcwd()}/../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
	from tensorflow.keras.applications.vgg16 import VGG16
	model_vg = VGG16(weights = 'imagenet', include_top = False)

	local_weights_file= path_vg

	pre_trained_model = VGG16(input_shape = (32, 32, 3),
                        include_top = False,
                        weights = None) 
	pre_trained_model.load_weights(local_weights_file)

		for layer in pre_trained_model.layers:
    			layer.trainable = False
   	 
	pre_trained_model.summary()
	last_layer = pre_trained_model.get_layer('block5_pool')
	print("Last layer Output shape:", last_layer.output_shape)
	last_output = last_layer.output

#Final layers
		from tensorflow.keras.optimizers import RMSprop
	x = layers.Flatten()(last_output)
	x = layers.Dense(1024, activation = 'relu')(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Dense(101, activation = 'softmax')(x)

	model = Model(pre_trained_model.input, x)

	model.compile(
        	optimizer = RMSprop(lr = 0.0001),
        	loss = 'categorical_crossentropy',
        	metrics = ['accuracy'])
	model.summary()

#101 directories LMeow 

	train_dir = '../input/food41/images'
	path_food = f"{getcwd()}/../input/food41/images"
	train_apple_pie_dir = '../input/food41/images/apple_pie'
	train_baby_back_ribs_dir = '../input/food41/images/baby_back_ribs'

	train_apple_pie_fnames = os.listdir(train_apple_pie_dir)
	train_baby_back_ribs_fnames = os.listdir(train_baby_back_ribs_dir)

	print(len(train_apple_pie_fnames))
	print(len(train_baby_back_ribs_fnames))

#Image Augmentation

	from tensorflow.keras.preprocessing.image import ImageDataGenerator
	train_datagen = ImageDataGenerator(rescale = 1.0/255.0,
                                    	   rotation_range = 40,
                                    	   width_shift_range = 0.2,
                                    	   height_shift_range = 0.2,
                                    	   shear_range = 0.2,
                                    	   zoom_range = 0.2,
                                    	   horizontal_flip = True)
	test_datagen = ImageDataGenerator(
    				rescale = 1.0/255.0
				)
	train_generator = train_datagen.flow_from_directory(
    					train_dir,
    					batch_size = 20,
    					class_mode = 'categorical',
    					target_size = (32, 32)
    					)
#test_generator = test_datagen.flow_from_directory(
 #   test_dir,
 #   batch_size = 20,
 #   class_mode = 'categorical',
 #   target_size = (32, 32)
 #)

	# Training 
	history = model.fit_generator(
           		 train_generator,
           	 	epochs = 5,
            		)