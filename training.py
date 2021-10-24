# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 12:58:22 2021
Mise en oeuvre d'une démarche de Deep Learning pour classifier des images
de couverts
@author: tom-h
"""

### Importing modules ########################################################
import cv2
import os
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


#fixing seed in order to save and reload models -> it still does not work
from numpy.random import seed
seed(42)
tf.random.set_seed(42)

### Defining functions #######################################################
def pngToJpg(folder):
    '''converts an png image to jpg'''
    for img in os.listdir(folder):
        if '.png' in img or '.PNG' in img:
            
            try:
                png_img = cv2.imread(folder+'//'+img)
                cv2.imwrite(f"folder+'//'+{img[:-4]}.jpg", png_img, 
                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                os.remove(folder+'//'+img)
                print(f'image convertie: {img[:-4]}.jpg')

            except Exception as e:
                print(e)
    
def make_model(input_shape):
    '''builds a CNN model, based on an input shape'''

    model = tf.keras.Sequential([
        
        data_augmentation,
        
        layers.Conv2D(6, 5, activation='relu', padding='same'),
        layers.AveragePooling2D((2,2)),

        layers.Conv2D(16, 5, activation='relu', padding='same'),
        layers.AveragePooling2D((2,2)),

        layers.Conv2D(32, 5, activation='relu', padding='same'),
        layers.AveragePooling2D((2,2)),
        layers.Dropout(0.5),
        
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.AveragePooling2D((2,2)),
        layers.Dropout(0.5),
        

        layers.Flatten(),
        
        layers.Dense(120, activation='relu'),        
        layers.Dense(84, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(2, activation='softmax')
        
        ])
    
    model.build(input_shape=input_shape)
    model.summary()

    return model


### Main Script ##############################################################

# conversion png to jpg ######################################################
folders_names = ['data//fork','data//spoon','data//knife']
for folder in folders_names:
    pngToJpg(folder)
    
##### generating datasets
image_size = (100,100)
batch_size = 80

input_shape = (batch_size,) + image_size + (1,)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data",
    validation_split=0.25,
    seed=123,
    subset="training",
    image_size=image_size,
    batch_size=batch_size,
    color_mode = 'grayscale'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data",
    validation_split=0.25,
    seed=123,
    subset="validation",
    image_size=image_size,
    batch_size=batch_size,
    color_mode = 'grayscale'
)

##### visualisation of the image data
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(2):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")

##### data augmentation layer 
data_augmentation = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.GaussianNoise(1),
        layers.experimental.preprocessing.RandomZoom(0.2),
        layers.experimental.preprocessing.RandomContrast(0.5)
    ]
)

##### visualisation of the data augmentation
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")

##### building the model
model = make_model(input_shape=input_shape)

#compiling the model
model.compile(optimizer='adam',
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

#fitting the model
epochs = 2000
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

#summarizing the training's history for accuracy
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.ylim(0,1)
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print(model.evaluate(val_ds))
#for some reason, the validation accuracy here is very different from the 
#one displayed by model.fit

#Confusion Matrix 
y_pred = model.predict(val_ds)
predicted_categories = tf.argmax(y_pred, axis=1)
true_categories = tf.concat([y for x, y in val_ds], axis=0)

predicted_categories = tf.cast(predicted_categories, tf.int32)
print(predicted_categories, len(predicted_categories))
print(true_categories, len(true_categories))
print(np.mean(predicted_categories==true_categories))

print(confusion_matrix(predicted_categories, true_categories))
