from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os


img = image.load_img(r"D:\projects\numbers_dataset\training\0\15.JPG")

train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory(r"D:\projects\numbers_dataset\training",
                                            target_size= (200,200),
                                            batch_size=32,
                                            class_mode='categorical')

validation_dataset = train.flow_from_directory(r"D:\projects\numbers_dataset\validation",
                                            target_size= (200,200),
                                            batch_size=32,
                                            class_mode='categorical')


""" print(train_dataset.class_indices)   #to check ur classes and their indices

print(train_dataset.classes) """

# M O D E L

model= tf.keras.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation="relu",input_shape =(200,200,3)),
                            tf.keras.layers.MaxPool2D(2,2),
                            #
                            tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
                            tf.keras.layers.MaxPool2D(2,2),
                            #
                            tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
                            tf.keras.layers.MaxPool2D(2,2),
                            #
                            tf.keras.layers.Flatten(),
                            #
                            tf.keras.layers.Dense(512,activation="relu"),
                            #
                            tf.keras.layers.Dense(10,activation="softmax")
                            ])    


model.compile(loss="binary_crossentropy",
            optimizer=RMSprop(lr=0.001),
            metrics=["accuracy"])

model.fit(train_dataset,
        steps_per_epoch=len(train_dataset),
        epochs=5,
        validation_batch_size=len(validation_dataset),
        validation_data=validation_dataset)

model.save(r'D:\projects\numbers_dataset\num_model.model')