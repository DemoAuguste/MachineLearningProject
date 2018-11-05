#!/usr/bin/python

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def get_data(filename):
	return np.loadtxt(filename)
 
labels = get_data('digits4000_txt/digits4000_digits_labels.txt')
digits = get_data('digits4000_txt/digits4000_digits_vec.txt')

# 两组数据
# 1. train 1-2000, test 2001-4000
# 2. train 2001-4000, test 1-2000
# 数据是按0-9的顺序排列的，每个数字有两百个
X_train = digits[:2000]
y_train = labels[:2000]
X_test = digits[2000:]
y_test = labels[2000:]
 
 
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)
print('-----predict value is ------')
print(y_predict)
print('-----actual value is -------')
print(y_test)
count = 0
for i in range(len(y_predict)):
	if y_predict[i] == y_test[i]:
		count += 1
print('accuracy is %0.2f%%'%(100*count/len(y_predict)))

