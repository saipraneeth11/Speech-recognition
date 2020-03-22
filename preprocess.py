# -*- coding: utf-8 -*-
"""


@author: SaiPr
"""
#librosa is useful is useful python library for speech processing 
import os 
import librosa 
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm 
#path is the location where data is present
PATH = 'path'

#getlabels tells about how many classes(words) are there
def getlabels(path=''):
    labels = os.listdir(path)
    labelindex = np.arange(0,len(labels))
    
    return labels , labelindex , to_categorical(labelindex)

#this is a method to convert a wav file to mfcc
    # mfcc stands Mel Frequency Cepstral Co-efficients.
def wavtomfcc(path, n_mfcc=20, max_len=11):
    wave, sr = librosa.load(path, mono=True, sr=None)
    wave = np.asfortranarray(wave[::3])
    mfcc = librosa.feature.mfcc(wave, sr=16000, n_mfcc=n_mfcc)

    
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')


    else:
        mfcc = mfcc[:, :max_len]
    
    return mfcc


#eachaudio is a method to convert all our wav files to mfcc and add a label to the wav file
def eachaudio (path , max_len =11 , n_mfcc = 20):
    labels , temp1, temp2 = getlabels(path)
    for label in labels :
        mfcc_vec = []
        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            mfcc = wavtomfcc(wavfile, max_len=max_len, n_mfcc=n_mfcc)
            mfcc_vec.append(mfcc)
        np.save(label + '.npy', mfcc_vec)
        
#This is basically for retriving the X_train ,Y_train .....
def get_train_test(split_ratio=0.6, random_state=42):
    # Get available labels
    labels, indices, _ = getlabels(PATH)

    # Getting first arrays
    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)


def prepare_dataset(path=PATH):
    labels, _, _ = getlabels(path)
    data = {}
    for label in labels:
        data[label] = {}
        data[label]['path'] = [path  + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]

        vectors = []

        for wavfile in data[label]['path']:
            wave, sr = librosa.load(wavfile, mono=True, sr=None)
            # Downsampling
            wave = wave[::3]
            mfcc = librosa.feature.mfcc(wave, sr=16000)
            vectors.append(mfcc)

        data[label]['mfcc'] = vectors

    return data


def load_dataset(path=PATH):
    data = prepare_dataset(path)

    dataset = []

    for key in data:
        for mfcc in data[key]['mfcc']:
            dataset.append((key, mfcc))

    return dataset[:100]