# coding: utf-8
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
np.random.seed(1337)
 
# download the mnist
#(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
def get_data(filename):
	return np.loadtxt(filename)
	
labels = get_data('digits4000_txt/digits4000_digits_labels.txt')
digits = get_data('digits4000_txt/digits4000_digits_vec.txt')

# 两组数据
# 1. train 1-2000, test 2001-4000
# 2. train 2001-4000, test 1-2000
# 数据是按0-9的顺序排列的，每个数字有两百个
X_train = digits[:2000]
Y_train = labels[:2000]
X_test = digits[2000:]
Y_test = labels[2000:]
 
# data pre-processing
X_train = X_train.reshape(-1, 1, 28, 28)/255
X_test = X_test.reshape(-1, 1, 28, 28)/255
Y_train = np_utils.to_categorical(Y_train, num_classes=10)
Y_test = np_utils.to_categorical(Y_test, num_classes=10)
 
# build CNN
model = Sequential()
 
# conv layer 1 output shape(32, 28, 28)
model.add(Convolution2D(filters=32,
		       kernel_size=5,
		       strides=1,
		       padding='same',
		       batch_input_shape=(None, 1, 28, 28),
		       data_format='channels_first'))
model.add(Activation('relu'))
 
# pooling layer1 (max pooling) output shape(32, 14, 14)
model.add(MaxPooling2D(pool_size=2, 
		       strides=2, 
		       padding='same', 
		       data_format='channels_first'))
 
# conv layer 2 output shape (64, 14, 14)
model.add(Convolution2D(64, 5, 
			strides=1, 
			padding='same', 
			data_format='channels_first'))
model.add(Activation('relu'))
 
# pooling layer 2 (max pooling) output shape (64, 7, 7)
model.add(MaxPooling2D(2, 2, 'same', 
		       data_format='channels_first'))
 
# full connected layer 1 input shape (64*7*7=3136), output shape (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
 
# full connected layer 2 to shape (10) for 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))
 
# define optimizer
adam = Adam(lr=1e-4)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
 
# training
print('Training')
model.fit(X_train, Y_train, epochs=10, batch_size=64)
 
# testing
print('Testing')
loss, accuracy = model.evaluate(X_test, Y_test)
print('loss, accuracy: ', (loss, accuracy))
 

