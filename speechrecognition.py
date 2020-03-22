# -*- coding: utf-8 -*-
"""


@author: SaiPr
"""
#preprocess is the python programm created
from preprocess import *
import keras
from keras.layers import Dense , Flatten , Conv2D , MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
import matplotlib.pyplot as plt

classes = 5  # number of diff words or classes
maxlen = 11 
buckets = 20

eachaudio('path', n_mfcc = buckets , max_len = maxlen )



X_train , X_test , Y_train , Y_test = get_train_test()

X_train = X_train.reshape(X_train.shape[0],buckets,maxlen,1)
X_test = X_test.reshape(X_test.shape[0],buckets,maxlen,1)

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

#A simple regression model 
#model = Sequential()
#model.add(Flatten(input_shape=(buckets, maxlen)))
#model.add(Dense(classes, activation='softmax'))
#model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
#model.fit(X_train,Y_train ,batch_size = 10 , nb_epoch=20)


#A CNN model   
# A deeper model helps in preventing overfiting              
classifier = Sequential()
classifier.add(Conv2D(32,(3,3),input_shape=(buckets,maxlen,1),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Conv2D(32,(3,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())
classifier.add(Dense(128,activation = 'relu'))
classifier.add(Dense(classes,activation = 'softmax'))
classifier.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])

#Final step traing the classifier
classifier.fit(X_train,Y_train ,batch_size = 10 , nb_epoch=20)

#The output predicted is a array of 5 elements
#array represents the probability of = ['five', 'four','one'..] alphabatically 

Y_pred = classifier.predict(X_test)






