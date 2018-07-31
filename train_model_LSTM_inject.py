#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 19:22:59 2018

@author: nikos
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage.transform import rescale, resize, downscale_local_mean
from lmfit.models import GaussianModel, ConstantModel
from keras.preprocessing import image
#from keras.applications.imagenet_utils import preprocess_input
from keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50

from keras.models import Model, load_model
from keras.layers import Input, Embedding, Dense, Activation, LSTM, GRU, Dropout, RepeatVector
from keras.layers.merge import concatenate
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model

from sklearn.model_selection import train_test_split
import random
import sys
import time
from keras.preprocessing.text import Tokenizer#map words to integers
from keras.backend import clear_session
#clear_session();print('Cleared Keras session to load new model')
import pickle

#%% load the data
filenames = np.load('Flickr8k_images_filenames.npy')#8091 filenames
images = np.load('Flickr8k_images_encoded.npy')#8091 images
captions = np.load('captions.npy').item()#5 captions per image
assert np.array_equal(np.sort(filenames),np.sort(np.array(list(captions.keys()))))

#%% Tokenize the captions: map each word/token to an integer
filenames_tr = pd.read_csv('./Flickr8k_text/Flickr_8k.trainImages.txt',header=None)
filenames_tr = np.array(filenames_tr.values.tolist())#convert to array with dtype='<U25'
captions_per_image=5

#find the training captions to fit the tokenizer on
captions_tr = list()
for f in filenames_tr:
    #captions_tr.append(captions[f[0]])
    captions_tr=captions_tr+captions[f[0]]
assert len(captions_tr) == len(filenames_tr)*captions_per_image
#max caption length in training data set
max_caption_length=max([len(x.split()) for x in captions_tr])
print('Maximum caption length:',max_caption_length,'words/tokens.')
#consider removing '.' from the filters
tokenizer = Tokenizer(num_words=None,filters='!"#$%&()*+,-./:;=?@[\]^_`{|}~',
                      lower=False, split=' ', char_level=False)
tokenizer.fit_on_texts(captions_tr)
vocab_size = len(tokenizer.word_index.keys())+1
print('Vocabulary size after tokenizer:',vocab_size,'unique words.')

#%% set up a generator function to train on one image at a time (conserve RAM)

def data_generator(input_filenames=None):
    '''
    Generate online training data, one image at a time.
    Note: one image produces several "datapoints", since every token of each
    caption is a different output target.
    Yields:
        X_img: (#timesteps,#imagefeatures):image feature input
        X_txt: (#timesteps,#max_caption_length):text input, each word is an integer
        y:     (#timesteps,#vocab_size):one-hot encoded output word to predict
    '''
    #filenames_gen = pd.read_csv(input_filepath,header=None)
    #filenames_gen = np.array(filenames_gen.values.tolist())#convert to array with dtype='<U25'
    #print('Generator for:',input_filepath)
    filenames_gen = input_filenames
    print('files total:',len(filenames_gen))
    while True:
        for f in filenames_gen:
            X_img, X_txt, y = list(), list(), list()#new list for every image
            ix = np.where(filenames==f)[0][0]#find the index of the image
            img = images[ix,:]#load the image features using the index
            img_captions = captions[f[0]]#load the captions of the image
            for c in img_captions:
                # encode the sequence
                seq = tokenizer.texts_to_sequences([c])[0]
                # split one sequence into multiple X,y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_caption_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)#[0]
                    # store
                    X_img.append(img)#append the image features
                    X_txt.append(in_seq)
                    y.append(out_seq)
            yield([[np.array(X_img), np.array(X_txt)], np.array(y)])  
    
#%% Specify the model
nembedding = 128
ndense = 128
nlstm = 128
dropout_rate=0.0
#dropout_rate=0.25
# feature extractor model
input_img = Input(shape=(2048,))
x_img = Dropout(dropout_rate)(input_img)
x_img = Dense(ndense, activation='relu')(x_img)
x_img = RepeatVector(max_caption_length)(x_img)#repeat in time

# sequence model
input_txt = Input(shape=(max_caption_length,))
x_txt = Embedding(vocab_size, nembedding, mask_zero=True)(input_txt)

x_merge = concatenate([x_img, x_txt])

x_merge = Dropout(dropout_rate)(x_merge)
x_merge = LSTM(nlstm)(x_merge)

# decoder model
x_merge = Dropout(dropout_rate)(x_merge)
x_merge = Dense(ndense, activation='relu')(x_merge)
#x_merge = Dropout(dropout_rate)(x_merge)
output = Dense(vocab_size, activation='softmax')(x_merge)
# tie it together [image, seq] [word]
model = Model(inputs=[input_img, input_txt], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# summarize model
print(model.summary())

#%% train the model
#generator for training data
filenames_tr = pd.read_csv('./Flickr8k_text/Flickr_8k.trainImages.txt',header=None)
filenames_tr = np.array(filenames_tr.values.tolist())#convert to array with dtype='<U25'
gen_train = data_generator(input_filenames=filenames_tr)
steps_per_epoch_tr =  len(filenames_tr)
#generator for validation data
filenames_val = pd.read_csv('./Flickr8k_text/Flickr_8k.devImages.txt',header=None)
filenames_val = np.array(filenames_val.values.tolist())#convert to array with dtype='<U25'
gen_val = data_generator(input_filenames=filenames_val)
steps_per_epoch_val = len(filenames_val)

filepath='./saved_models/model128_LSTM_inject_dropout'+str(dropout_rate) #to save the weights
#save model architecture as a .png file
plot_model(model, to_file=filepath+'.png', show_shapes=True)
#save tokenizer to use on new datasets
with open(filepath+'_tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
##how to load the tokenizer
#with open('tokenizer.pkl', 'rb') as handle:
#    tokenizer = pickle.load(handle)

checkpoint = ModelCheckpoint(filepath+'.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
csvlog = CSVLogger(filepath+'_train_log.csv',append=True)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)

tic=time.time()
model.fit_generator(generator=gen_train,steps_per_epoch=steps_per_epoch_tr,
                    validation_data=gen_val,validation_steps=steps_per_epoch_val,
                 epochs=10, verbose=2,
                 initial_epoch=0,callbacks=[checkpoint, csvlog, early_stopping])
toc=time.time()
model.save(filepath+'_model.hdf5')
file = open(filepath+'_time.txt','w')
file.write('training time:'+format(toc-tic, '.2f')+'seconds')
file.close()









































