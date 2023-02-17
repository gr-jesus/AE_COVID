import cv2
import random
import numpy as np
import SimpleITK as sitk
from os import walk
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

input_img=Input(shape=( 128, 128, 4), name='Input')
x=Convolution2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', name='Conv1E')(input_img)
x=MaxPooling2D((2,2), name='Maxpool1E')(x)
x=Convolution2D(64,(3,3), activation='relu', padding='same', name='Conv2E')(x)
x=MaxPooling2D((2,2), name='Maxpool2E')(x)
x=Convolution2D(32,(3,3), activation='relu', padding='same', name='Conv3E')(x)
x=MaxPooling2D((2,2), name='Maxpool3E')(x)
x=Convolution2D(16,(3,3), activation='relu', padding='same', name='Conv4E')(x)
encoded=MaxPooling2D((2,2), name='Maxpool4E')(x)
x=Convolution2D(32,(3,3), activation='relu', padding='same', name='Conv1D')(encoded)
x=UpSampling2D((2,2), name='Maxpool1D')(x)
x=Convolution2D(32,(3,3), activation='relu', padding='same', name='Conv2D')(x)
x=UpSampling2D((2,2), name='Maxpool2D')(x)
x=Convolution2D(64,(3,3), activation='relu', padding='same', name='Conv3D')(x)
x=UpSampling2D((2,2), name='Maxpool3D')(x)
x=Convolution2D(128,(3,3), activation='relu', padding='same', name='Conv4D')(x)
x=UpSampling2D((2,2), name='Maxpool4D')(x)
decoded=Convolution2D(4,(1,1), activation='sigmoid', name='Decode')(x)
autoencoder=Model(input_img, decoded)
print(autoencoder.summary())

plot_model(autoencoder, to_file='model_plot.png', show_shapes=True, show_layer_names=True)