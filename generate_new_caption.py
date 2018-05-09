# -*- coding: utf-8 -*-
"""
Created on Tue May  8 11:03:08 2018

@author: N.Chlis
"""

import matplotlib
matplotlib.use('Agg')#don't show plots
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
from keras.layers import Input, Embedding, Dense, Activation, LSTM, GRU, Dropout
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
from nltk.translate.bleu_score import corpus_bleu #BLEU score

#%% helper functions and constants

#### DO NOT CHANGE THESE PARAMETERS ########################
target_height = 224#image height                           #
target_width = 224#image width                             #
max_caption_length = 39#computed in train_model.py         #
P = 2048#model.output_shape[-1]#2048 features from ResNet50#
############################################################

# map an integer to a word
'''
integer: the input id
tokenizer: the (fitted) tokenizer
'''
def index_to_token(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate a single caption for an image
def generate_caption(img, model, tokenizer, max_caption_length):
    '''
    img: the input image used to generate the captions
    tokenizer: a (fitted) tokenizer
    max_caption_length: the maximum number of words/tokens to produce
    '''
    # seed the generation process
    in_text = '<START>'
    img = img.reshape((1,)+img.shape)
    # iterate over the whole length of the sequence
    for i in range(max_caption_length):
        # encode each text sequence to an integer sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input to max_caption_length size
        sequence = pad_sequences([sequence], maxlen=max_caption_length)
        #sequence = sequence.reshape((1,)+sequence.shape)
        # predict next word/token
        next_id = model.predict([img,sequence], verbose=0)
        # select the class (integer index) with the highest probability
        next_id = np.argmax(next_id)
        # map integer index to word
        word = index_to_token(next_id, tokenizer)
        #print('next word',word)
        # stop if we cannot map the word
        if word is None:
            #print('found None, exiting')
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == '<END>':
            break
    return in_text

#%% change these parameters as you see fit
img_folder = './new_images'#folder containing the images to caption
save_folder = './captioned_images'#where to save captioned images, make sure to create this folder before running the script
captioning_model = './saved_models/model128_GRU_dropout0.25.hdf5'#keras model

#%% read all images in the folder, generate captions and save them

#% load the filenames of all images
img_filenames = sorted(os.listdir(img_folder))#sort to alphabetical order
img_filenames = np.array(img_filenames)
N = len(img_filenames)#number of total images

#load the tokenizer
with open('./saved_models/model128_GRU_dropout0.25_tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
tokenizer.oov_token = None #attribute lost during serialization
vocab_size = len(tokenizer.word_index.keys())+1
print('Vocabulary size after tokenizer:',vocab_size,'unique words.')

#encode and caption each image
i=0
for f in img_filenames:
    print('captioning image',i+1,'of',N)
    #print(f)
    #load and resize at the same time
    img = image.load_img(os.path.join(img_folder,f),
                         target_size=(target_height,target_width))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)#include N_images dimension
    x = preprocess_input(x)#so it's ready for pre-trained imagenet CNN
    
    #load the CNN and encode the image
    clear_session()#to avoid keras breakdown
    model = ResNet50(weights='imagenet',include_top=False)#load the CNN
    X_enc=model.predict(x).reshape((1,P))#encode the image
    
    #load captioning model, perform sanity checks and generate caption
    clear_session()#to avoid keras breakdown
    model=load_model(captioning_model)
    assert model.input_layers[1].input_shape[1] == max_caption_length
    assert model.output_layers[0].output_shape[1] == vocab_size
    caption = generate_caption(X_enc[0,:],model=model,tokenizer=tokenizer,
                               max_caption_length=max_caption_length)
    
    #save the image and the caption
    plt.imshow(img)
    plt.xticks([])#hide x-axis ticks
    plt.yticks([])#hide y-axis ticks
    plt.title(caption)
    plt.savefig(save_folder+'/captioned_'+f,dpi=100,bbox_inches='tight')
    
    i=i+1
