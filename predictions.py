# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 22:41:20 2021
Prediction of new image data by the model
@author: tom-h
"""
### importing modules ########################################################
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image

### defining functions #######################################################
def capture_webcam_images():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    img_counter = 0
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)
    
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "test_webcam//frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
    
    cam.release()
    cv2.destroyAllWindows()
    
def get_label_name(int_):
    '''get the class label from the int encoding'''
    if int_ == 0:
        name = 'fork'
    elif int_ == 1:
        name = 'knife'
    else:
        name = 'spoon'
    return name

def testing(model, img_dir, image_size):
    '''tests the predictions of a model based on image data'''
    for i,img in enumerate(os.listdir(img_dir)):
      img = image.load_img(os.path.join(img_dir,img), 
                            target_size=image_size, color_mode='grayscale')
      img2 = np.reshape(img, (1, 100, 100, 1))
      plt.imshow(img)
      img = np.expand_dims(img, axis=0)
      
      result=model.predict_classes(img2)
      plt.title(get_label_name(result[0]))
      print('---Image 1')
      print(result)
      plt.show()
      
      input('Tapez entrée pour continuer.. ')

### main script ##############################################################

#importing pre-built model
model = tf.keras.models.load_model('modele_final')
image_size = (100,100)
img_dir='test'

# testing(model, img_dir, image_size)

#testing captures from webcam
img_dir_webcam = 'test_webcam'

capture_webcam_images()
testing(model, img_dir_webcam, image_size)







