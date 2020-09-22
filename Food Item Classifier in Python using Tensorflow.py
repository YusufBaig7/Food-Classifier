import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model  

batch_size = 100
img_h = 128
img_w = 128
num_clas=100

clas = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap',
           'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
           'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla','chicken_wings','chocolate_cake',
           'chocolate_mousse','churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame',
           'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon'
           'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
           'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole',
           'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque',
           'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos','omelette', 'onion_rings',
           'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine',
           'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops',
           'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
           'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles' ]

train_datagen = ImageDataGenerator(rotation_range=50,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    zoom_range=0.3,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode='constant',
                                    cval=0,
                                    rescale=1./255)
valid_datagen = ImageDataGenerator(rotation_range=45,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    zoom_range=0.3,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode='constant',
                                    cval=0,
                                    rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = '../input/food101/training'
val_dir = '../input/food101/validation' 
seeed = 10
train_gen = train_datagen.flow_from_directory(train_dir,
                                              batch_size = batch_size,
                                              target_size = (128,128),
                                              classes = clas,
                                              class_mode = 'categorical',
                                              shuffle = True,
                                              seed = seeed)
valid_gen = valid_datagen.flow_from_directory(val_dir,
                                              batch_size = batch_size,
                                              target_size = (128,128),
                                              classes = clas,
                                              class_mode = 'categorical',
                                              shuffle = True,
                                              seed = seeed)


from keras.backend import sigmoid
def swish(x, beta =1):
    return(x*sigmoid(beta*x))

from tensorflow.keras import regularizers
from tensorflow.keras.layers import GlobalAveragePooling2D,Flatten, Dense, Dropout
ResNet_model = tf.keras.applications.ResNet152V2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# The last 15 layers fine tune
for layer in ResNet_model.layers[:-15]:
    layer.trainable = False

x = ResNet_model.output
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
x = Dense(units=512, activation='swish')(x)
x = Dropout(0.3)(x)
x = Dense(units=512, activation='swish')(x)
x = Dropout(0.3)(x)
output  = Dense(units=100, activation='softmax')(x)
model = Model(ResNet_model.input, output)


model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import ReduceLROnPlateau

learn_rate = ReduceLROnPlateau(monitor='val_accuracy', 
                        patience=3, 
                        verbose=1, 
                        factor=0.4, 
                        min_lr=0.0001)


callbacks = [learn_rate]

STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
STEP_SIZE_VALID=valid_gen.n//valid_gen.batch_size
history = model.fit_generator(generator=train_gen,
                   steps_per_epoch=STEP_SIZE_TRAIN,
                   validation_data=valid_gen,
                   validation_steps=STEP_SIZE_VALID,
                   epochs=50,
                   callbacks=callbacks)

model.evaluate_generator(generator=valid_gen,steps=STEP_SIZE_VALID)

%matplotlib inline
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


#### Predict function
def predict_food(imagepath):
    image = cv2.imread(imagepath)
    res = cv2.resize(image, dsize = (150, 150),interpolation=cv2.INTER_AREA)
    res = np.array(res)
    res  = np.expand_dims(res, axis = 0)
    predict = model.predict(res)
    return predict

print (predict_food('Your file directory'))


