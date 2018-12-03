#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 16:38:46 2018

@author: wangruohan
"""
import numpy as np
from sklearn.decomposition import pca
import time
from sklearn import metrics
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.optimizers import RMSprop

def knn_classifier(train_x,train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model

def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model

def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model

def svm_classifier_rbf(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', gamma=0.01, C=1) # 选择kernel之后，rbf的准确率比其他的高，参数还没研究过
    model.fit(train_x, train_y)
    return model

def svm_classifier_linear(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='linear', gamma=0.01, C=1) 
    model.fit(train_x, train_y)
    return model


def random_forest_classifier(train_x, train_y, num_classes=10):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(train_x, train_y)
    return model

def ada_boost_classifier(train_x, train_y, num_classes=10):
	from sklearn.ensemble import AdaBoostClassifier
	model = AdaBoostClassifier()
	model.fit(train_x,train_y)
	return model

def cnn_classifier(train_x, train_y):
    train_X = train_x.reshape(-1, 1, 28, 28)
    train_Y = np_utils.to_categorical(train_y, num_classes=10)
    
    model = Sequential()
    
    model.add(Convolution2D(filters=32,
                                        kernel_size=5,
                                        strides=1,
                                        padding='same',
                                        batch_input_shape=(None, 1, 28, 28),
                                        data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2, 
                                       strides=2, 
                                       padding='same', 
                                       data_format='channels_first'))
    model.add(Convolution2D(64, 5, 
                                        strides=1, 
                                        padding='same', 
                                        data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))
    
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    adam = Adam(lr=1e-4)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(train_X, train_Y, epochs=20, batch_size=64)
    
    return model

def dnn_classifier(train_x, train_y):
    batch_size = 100
    nb_classes = 10
    nb_epoch = 20
    
    train_y = np_utils.to_categorical(train_y, num_classes=10)
    
    input_dim = train_x.shape[1]
    
    model = Sequential()
    model.add(Dense(512, input_shape=(input_dim,)))
    
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(512))
    model.add(Activation('relu'))
    
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    model.fit(train_x, train_y, batch_size=batch_size, epochs=nb_epoch)
    
    return model