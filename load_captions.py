#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 22:40:04 2018

@author: nikos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string

#df=pd.read_csv('./Flickr8k_text/Flickr_8k.devImages.txt',header=None)
#fnames=np.array(df[0].values.tolist())#to get as dtype('<U25'), otherwise it's object

#%% load all captions from a .txt file

#filename = './Flickr8k_text/Flickr8k.token.txt'
filename = './Flickr8k_text/Flickr8k.token_clean.txt'#removed image 2258277193_586949ec62.jpg that doesn't exist in data folder
#read file containing the descriptions
file = open(filename, 'r')
text = file.read()#entire file read as one string 
file.close()

#dictionary of {'image_id0': ['caption0',...,'captionN'],
#               'image_id1': ['caption0',...,'captionN'],...}
captions = dict()

#for each line of the string find the image id and captions
#preprocess every caption (e.g. lowercase) and then save it
text_lines = text.split('\n')
#line=text_lines[0]
for line in text_lines:
    if(len(line))>0:#lines[-1]=='' with len('')==0
        #split line by white space
        tokens = line.split()
        #take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        #pre-process the image_id
        image_id = image_id.split('#')[0]# lose the #* part e.g. #0, #1 etc.
        #image_id = image_id.split('.jpg')[0]# lose the filetype
        #convert description tokens back to string
        image_desc = ' '.join(image_desc)
        #pre-process the descriptions
        image_desc = image_desc.lower()#convert to lowercase
        #image_desc = image_desc.translate(str.maketrans('','', string.punctuation))#remove punctuation
        image_desc = image_desc.strip()#remove any whitespace before/after
        image_desc = image_desc.strip(' .')#remove ' .' before/after
        #add <START> and <END> tokens
        image_desc = '<START> '+image_desc+' <END>'
        #create a new dictionary key for a new image (unseen) image_id
        if image_id not in captions:
            captions[image_id] = list()
        # store description
        captions[image_id].append(image_desc)

#sanity check: all images must have the same number of captions (five)
captions_per_image=5
l=[] #list of #captions per image
for v in captions.values():
    l.append(len(v))
assert np.unique(l) == captions_per_image

#Vocabulary size: how many unique words are present
words = set()
for key in captions.keys():
    [words.update(c.split()) for c in captions[key]]
print('All captions loaded successfully')
print('Total images:',len(captions))
print('Total captions:',len(captions)*captions_per_image,'-> '+
      str(captions_per_image)+' captions per image')
print('Vocabulary size:',len(words),'unique words.')
#example_img=list(captions.keys())[0]
example_img = image_id#last image that was loaded
print('\nExample captions for image ' +example_img+':')
for v in captions[example_img]:
    print(v)

#%% save the dictionary using numpy
#np.save("captions.npy", captions, allow_pickle = True)

#%% sanity check: load saved captions and compare
captions2 = np.load('captions.npy').item()#.item() necessary for structured array

assert type(captions2)==dict #it is read as a dictionary
assert len(captions) == len(captions2)#they must have the same length
#they must also have the same values for each key
for key in captions.keys():
    #print(key)
    assert captions[key]==captions2[key]
print('\nAll captions saved successfully.')
    





















