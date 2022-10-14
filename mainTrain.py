import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from numpy import append
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

#Importing Images

image_directory='datasets/'

no_dementia_images=os.listdir(image_directory+ 'no/')
yes_dementia_images=os.listdir(image_directory+ 'yes/')

dataset=[]                                                                      #list
label=[]                                                                        #list
INPUT_SIZE =128
#print(no_dementia_images)

#path='26 (100).jpg'

#print(path.split('.')[1])

#pre processing images

for i, image_name in enumerate(no_dementia_images):
    if (image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+ 'no/'+ image_name)
        image=Image.fromarray(image, 'RGB')                                     #Converting to RGB
        image=image.resize((INPUT_SIZE,INPUT_SIZE))                             #Resizing the images
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(yes_dementia_images):
    if (image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+ 'yes/'+ image_name)
        image=Image.fromarray(image, 'RGB')                                     #Converting to RGB
        image=image.resize((INPUT_SIZE,INPUT_SIZE))                             #Resizing the images
        dataset.append(np.array(image))
        label.append(1)

#Train Test Split

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

#Reshape = (n, image_width, image_height, n_channel)

#print(x_train.shape)
#print(x_test.shape)
#print(y_train.shape)
#print(y_test.shape)

x_train=normalize(x_train, axis=1)
x_test=normalize(x_test, axis=1)

#Model Building

model = Sequential()

model.add(Conv2D(32,(3,3), input_shape=(INPUT_SIZE,INPUT_SIZE,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.003)

history=model.fit(x_train, y_train, batch_size=16, verbose=1, callbacks=[early_stopping], epochs=50, validation_data=(x_test,y_test),shuffle=False)

model.save('Dementia50Epochs128early.h5')

#Graphical Representation

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs Validation Loss')
plt.ylabel('Loss and Validation Loss')
plt.xlabel('Epoch')
plt.legend(['Loss','Validation loss'], loc='upper right')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy vs Validation Accuracy')
plt.ylabel('Accuracy and Validation Accuracy')
plt.xlabel('Epoch')
plt.legend(['accuracy','validation accuracy'], loc='lower right')
plt.show()


