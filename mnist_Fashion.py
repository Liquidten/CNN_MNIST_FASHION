#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 00:21:41 2019

@author: sameepshah
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

print(tf.__version__)


def Model(train_images,train_labels, input_shape):
    
    model = keras.Sequential([
            keras.layers.Conv2D(28, kernel_size=(3,3),activation='relu', input_shape=(input_shape)),
            keras.layers.Conv2D(64, (3,3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size = (2,2)),
            #Randomely dropping neurons to improve convergence
            keras.layers.Dropout(0.2),
            #Flatten to reduce dimentions
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation=tf.nn.softmax)
            ])
    
    return model

def plot_image(i, predictions_array, true_label, img, class_names):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img.squeeze(), cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    if predicted_label ==  true_label:
        color = 'green'
    else:
        color = 'red'
    
    
    plt.xlabel("{} {:2.0f}% ({}))".format(class_names[predicted_label],100*np.max(predictions_array), class_names[true_label]), color=color)
    
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')
    
    
def main():   
    
    #import data
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images.shape
    len(train_labels)
    #train_labels
    test_images.shape
    len(test_labels)

    plt.figure()
    plt.imshow(train_images[1])
    plt.colorbar()
    plt.grid(False)
    plt.show
    
    
    #Reshaping the array to 4 dims so that it can work with Kears API
    
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    
    #making sure that the values are float so that we can get decimal points after division
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')
    
    #normalizing the RGB codes by dividing it to the max RGB
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    #print("train_images: ", train_images)
    print("Number of images in train_images:", train_images.shape[0])
    print("Number of images in test_images:", test_images.shape[0])
    
    
    
    plt.figure(figsize=(10,10))
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i].squeeze(), cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()
    
    
    model = Model(train_images,train_labels, input_shape)
    model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy',
                  metrics = ['accuracy'])
    model_log = model.fit(train_images, train_labels, epochs = 30, batch_size = 64, validation_split=0.2)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)
    
    predictions = model.predict(test_images)
    #print(predictions[2])
    np.argmax(predictions[2])
    #print(test_labels[2])
    
   
    #plotting the metrics
    fig = plt.figure()
    
    plt.subplot(2,1,1)
    plt.plot(model_log.history['acc'])
    plt.plot(model_log.history['val_acc'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc ='upper right')
    
    
    plt.sublpot(2,1,2)
    plt.plot(model_log.history['loss'])
    plt.plot(model_log.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    
    plt.tight_layout()
    fig
    
    i = 0
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions, test_labels, test_images, class_names)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions, test_labels)
    plt.show()
    
    i = 10
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions, test_labels, test_images, class_names)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions,  test_labels)
    plt.show()
    
    
    #Save the model
    #serialize model to JSON
    model_digit_json = model.to_json()
    with open("model_digit.json", "w") as json_file:
        json_file.write(model_digit_json)
    #serialize weights to HDF5
    model.save_weights("model_digit.h5")
    print("Saved model to disk")
    
    #plot the firs X test images, their predicted label, and the true label
    num_rows =5
    num_cols =3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions, test_labels, test_images, class_names)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions, test_labels)
    plt.show()
    
    #Using the traing model to make a prediciton about a single image
    img =test_images[0]
    #print(img.shape)
    img = (np.expand_dims(img,0))
    #print(img.shape)
    predictions_single = model.predict(img)
    #print(predictions_single)
    plot_value_array(0, predictions_single, test_labels)
    _= plt.xticks(range(10), class_names, rotation=45)
    np.argmax(predictions_single[0])
    
main()


