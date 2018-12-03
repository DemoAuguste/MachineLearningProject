#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 16:42:07 2018

@author: wangruohan
"""

import numpy as np
from sklearn.decomposition import pca
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import FastICA
import time
from sklearn import metrics
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from sklearn.preprocessing import StandardScaler

def PCA_model(train_x,components=150):
    model = pca.PCA(n_components=components).fit(train_x)
    return model

def scale_model(train_x):
    model = StandardScaler().fit(train_x)
    return model

def KPCA_model(train_x,components):
    model = KernelPCA(n_components=components,kernel = 'linear').fit(train_x)
    return model

def fac_model(train_x,components):
    model = FactorAnalysis(n_components=components).fit(train_x)
    return model

def ica_model(train_x,components):
    model = FastICA(n_components=components).fit(train_x)
    return model

