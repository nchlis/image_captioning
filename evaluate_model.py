# -*- coding: utf-8 -*-
"""
Created on Tue May  8 11:03:08 2018

@author: N.Chlis
"""

import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.backend import clear_session
import pickle
from nltk.translate.bleu_score import corpus_bleu #BLEU score
from keras.models import load_model

#%% load the data
filenames = np.load('Flickr8k_images_filenames.npy')#8091 filenames
images = np.load('Flickr8k_images_encoded.npy')#8091 images
captions = np.load('captions.npy').item()#5 captions per image
assert np.array_equal(np.sort(filenames),np.sort(np.array(list(captions.keys()))))
max_caption_length = 39#computed in train_model.py

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

# evaluate the generated captions of the model on a
# set of images, using ground truth captions
def evaluate_model(model, filenames_eval, images, tokenizer, max_caption_length):
    '''
    model: a (trained) keras model
    filenames_eval: a list of filenames of images to evaluate on
    tokenizer: a (fitted) tokenizer
    max_caption_length: the maximum number of words/tokens to produce
    '''
    actual, predicted = list(), list()
    # step over the whole set
    for f in filenames_eval:
        #print('filename',f)
        ix = np.where(filenames==f)[0][0]#find the index of the image
        img = images[ix,:]#load the image features using the index
        img_captions = captions[f[0]]#load the captions of the image
        # generate description
        caption_predicted = generate_caption(img,model=model,
                                             tokenizer=tokenizer,
                                             max_caption_length=max_caption_length)
        # store actual and predicted
        captions_truth = [d.split() for d in img_captions]
        actual.append(captions_truth)
        predicted.append(caption_predicted.split())
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

#%% evaluate the models
filenames_ts = pd.read_csv('./Flickr8k_text/Flickr_8k.testImages.txt',header=None)
filenames_ts = np.array(filenames_ts.values.tolist())#convert to array with dtype='<U25'

model_filenames = ['model128_LSTM_dropout0.0','model128_LSTM_dropout0.1','model128_LSTM_dropout0.25',
                   'model128_GRU_dropout0.0','model128_GRU_dropout0.1','model128_GRU_dropout0.25'
                   'model128_LSTM_inject_dropout0.0','model128_LSTM_inject_dropout0.1','model128_LSTM_inject_dropout0.25',
                   'model128_GRU_inject_dropout0.0','model128_GRU_inject_dropout0.1','model128_GRU_inject_dropout0.25']
for mfname in model_filenames:
    clear_session()#to avoid keras breakdown
    print('Loading',mfname)
    
    #load the tokenizer
    with open('./saved_models/'+mfname+'_tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    tokenizer.oov_token = None #attribute lost during serialization
    vocab_size = len(tokenizer.word_index.keys())+1
    print('Vocabulary size after tokenizer:',vocab_size,'unique words.')
    
    #load model and perform sanity checks
    model=load_model('./saved_models/'+mfname+'.hdf5')
    assert model.input_layers[1].input_shape[1] == max_caption_length
    assert model.output_layers[0].output_shape[1] == vocab_size
    
    evaluate_model(model,filenames_ts,images,tokenizer,max_caption_length)

