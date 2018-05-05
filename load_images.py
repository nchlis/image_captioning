#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 15:21:26 2018

@author: nikos
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import cv2
import h5py
import skimage
from skimage.transform import rescale, resize, downscale_local_mean
from lmfit.models import GaussianModel, ConstantModel
import keras
from keras.preprocessing import image
#from keras.applications.imagenet_utils import preprocess_input
from keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50
#from keras.applications.vgg16 import VGG16

#%% load and encode each image on-the-fly
img_folder = './Flickr8k_Dataset'
#img_folder = './Flickr8k_Dataset_small'
target_height = 224
target_width = 224
model = ResNet50(weights='imagenet',include_top=False)
#def load_images(filepath=img_folder,target_height=target_height,target_width=target_width,
#                  normalize=True, subsample = False, subsample_rate = 8,
#                  save_to_file = False, save_filename = 'patient_images.h5'):

#% load the filenames of all images
img_filenames = sorted(os.listdir(img_folder))#sort to alphabetical order
img_filenames = np.array(img_filenames)
N = len(img_filenames)#number of total images
P = 2048#model.output_shape[-1]#2048 features from ResNet50
#initialize to N x Height x Width x Channels
X_enc = np.zeros(shape=(N,P))

#load each image, preprocess for imagenet, encode and save into X_enc
i=0
for f in img_filenames:
    print('loading image',i+1,'of',N)
    #load and resize at the same time
    img = image.load_img(os.path.join(img_folder,f),
                         target_size=(target_height,target_width))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)#include N_images dimension
    x = preprocess_input(x)#so it's ready for pre-trained imagenet CNN
    X_enc[i,:]=model.predict(x)#encode
    i=i+1
    #plt.imshow(x[0,:,:,:])

#%%
np.save('Flickr8k_images_encoded.npy',X_enc,allow_pickle=False)
np.save('Flickr8k_images_filenames.npy',img_filenames,allow_pickle=False)