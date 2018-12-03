#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 15:36:40 2018

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
from models import knn_classifier, logistic_regression_classifier, decision_tree_classifier, svm_classifier_rbf, svm_classifier_linear, random_forest_classifier, ada_boost_classifier, cnn_classifier, dnn_classifier
import preprocess
from sklearn.preprocessing import PolynomialFeatures

def load_data(flag=1,normalize=False):
    """
    flag: 选择两组数据中的某一个
    normalize: 数据是否要正则化 (在训练某些model的时候好像有bug，待研究)
    
    return: X_train, Y_train, X_test, Y_test
    """
    labels = get_data('digits4000_txt/digits4000_txt/digits4000_digits_labels.txt')
    digits = get_data('digits4000_txt/digits4000_txt/digits4000_digits_vec.txt')

    if flag==1:
        X_train = digits[:2000]
        Y_train = labels[:2000]
        X_test = digits[2000:]
        Y_test = labels[2000:]
    else:
        X_train = digits[2000:]
        Y_train = labels[2000:]
        X_test = digits[:2000]
        Y_test = labels[:2000]

    if normalize:
        X_train = X_train/255
        Y_train = Y_train
        X_test = X_test/255
        Y_test = Y_test

    return np.array(X_train),np.array(Y_train),np.array(X_test),np.array(Y_test)

def get_data(filename):
    return np.loadtxt(filename)

def training_process(classifiers, x_train, y_train, x_test, y_test):
    result = {}
    for classifier in classifiers:
        try:
            print("=======================")
            print('Classifier: {}'.format(classifier))
            start_time = time.time()
            temp_model = eval(classifier)(x_train,y_train)

            if classifier == 'cnn_classifier': # CNN需要转换一下数据格式
                x_test_reshape = x_test.reshape(-1, 1, 28, 28)
                y_test_reshape = np_utils.to_categorical(y_test, num_classes=10)
                loss, accuracy = temp_model.evaluate(x_test_reshape, y_test_reshape)
            else:
                if classifier == 'dnn_classifier':
                    y_test_reshape = np_utils.to_categorical(y_test, num_classes=10)
                    loss, accuracy = temp_model.evaluate(x_test, y_test_reshape)
                else:
                    y_train_predict = temp_model.predict(x_train)
                    training_accuracy = metrics.accuracy_score(y_train,y_train_predict)
                    print('training accuracy: {}'.format(training_accuracy))
                    y_predict = temp_model.predict(x_test)
                    accuracy = metrics.accuracy_score(y_test, y_predict)  
            print('testing accuracy: {}'.format(accuracy))
            print('training took %fs.'%(time.time()-start_time))
            result[classifier] = accuracy
        except:
            print('+++++++++++++++++++++++++')
            print('Error with {}.'.format(classifier))
            import traceback
            print(traceback.format_exc())
            print('+++++++++++++++++++++++++')
    return result

def main():

    classifiers = ['knn_classifier','logistic_regression_classifier','decision_tree_classifier','svm_classifier_rbf', 'svm_classifier_linear', 'random_forest_classifier','ada_boost_classifier','cnn_classifier','dnn_classifier']

    x_train, y_train, x_test, y_test = load_data(flag=2,normalize=True)
    

    """
    #没有PCA
    training_process(classifiers, x_train, y_train, x_test, y_test)
    """
    """
    #有PCA
    pca_model = preprocess.PCA_model(x_train)
    x_train_pca = pca_model.transform(x_train)
    x_test_pca = pca_model.transform(x_test)
    temp_classifiers = ['knn_classifier','logistic_regression_classifier','decision_tree_classifier','svm_classifier_rbf', 'svm_classifier_linear','random_forest_classifier','ada_boost_classifier','dnn_classifier'] # 没有CNN
    training_process(temp_classifiers, x_train_pca, y_train, x_test_pca, y_test)
    """ 
    """
    #多项式特征
    poly = PolynomialFeatures(2)
    print(x_train.shape)
    x_train = poly.fit_transform(x_train)
    print(x_train.shape)
    x_test = poly.fit_transform(x_test)
    pca_model = preprocess.PCA_model(x_train)
    x_train_pca = pca_model.transform(x_train)
    x_test_pca = pca_model.transform(x_test)
    temp_classifiers = ['knn_classifier','logistic_regression_classifier','decision_tree_classifier','svm_classifier_rbf', 'svm_classifier_linear','random_forest_classifier','ada_boost_classifier','dnn_classifier'] # 没有CNN
    training_process(temp_classifiers, x_train_pca, y_train, x_test_pca, y_test)
    """
    """
    #标准化
    scaler = preprocess.scale_model(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    training_process(classifiers, x_train, y_train, x_test, y_test)
    """
    """
    #kernel PCA
    kpca_model=preprocess.KPCA_model(x_train,components=600)
    x_train_pca = kpca_model.transform(x_train)
    x_test_pca = kpca_model.transform(x_test)
    temp_classifiers = ['knn_classifier','logistic_regression_classifier','decision_tree_classifier','svm_classifier_rbf', 'svm_classifier_linear','random_forest_classifier','ada_boost_classifier','dnn_classifier'] # 没有CNN
    training_process(temp_classifiers, x_train_pca, y_train, x_test_pca, y_test)
    """
    """
    #FactorAnalysis
    factor_model=preprocess.fac_model(x_train,components=6150)
    x_train_fac = factor_model.transform(x_train)
    x_test_fac = factor_model.transform(x_test)
    temp_classifiers = ['knn_classifier','logistic_regression_classifier','decision_tree_classifier','svm_classifier_rbf', 'svm_classifier_linear','random_forest_classifier','ada_boost_classifier','dnn_classifier'] # 没有CNN
    training_process(temp_classifiers, x_train_fac, y_train, x_test_fac, y_test)
    """
    """
    #ICA
    ICA_model = preprocess.ica_model(x_train,components=50)
    x_train_ica = ICA_model.transform(x_train)
    x_test_ica = ICA_model.transform(x_test)
    temp_classifiers = ['knn_classifier','logistic_regression_classifier','decision_tree_classifier','svm_classifier_rbf', 'svm_classifier_linear','random_forest_classifier','ada_boost_classifier','dnn_classifier'] # 没有CNN
    training_process(temp_classifiers, x_train_ica, y_train, x_test_ica, y_test)
    
    """
    #PCA+ICA
    pca_model = preprocess.PCA_model(x_train)
    x_train_pca = pca_model.transform(x_train)
    x_test_pca = pca_model.transform(x_test)
    ICA_model = preprocess.ica_model(x_train_pca,components=50)
    x_train_ica = ICA_model.transform(x_train_pca)
    x_test_ica = ICA_model.transform(x_test_pca)
    temp_classifiers = ['knn_classifier','logistic_regression_classifier','decision_tree_classifier','svm_classifier_rbf', 'svm_classifier_linear','random_forest_classifier','ada_boost_classifier','dnn_classifier'] # 没有CNN
    training_process(temp_classifiers, x_train_ica, y_train, x_test_ica, y_test)
    
    
if __name__ == '__main__':
    main()





