#!/usr/bin/python

from load_data import load_data
from models import *
import numpy as np
from sklearn import metrics
import copy

x_train,y_train,x_test,y_test = load_data(normalize=True)
classifiers = ['knn_classifier','logistic_regression_classifier','decision_tree_classifier','svm_classifier','cnn_classifier','dnn_classifier']

def binary_training(model,one_hot=False):
	binary_trained_model = {}
	for i in range(10):
		for j in range(i+1,10):
#			print('training {},{}'.format(i,j))
			temp_train_x = np.row_stack((x_train[200*i:200*(i+1)],x_train[200*j:200*(j+1)]))
			if one_hot:
				label_1 = np.array([[1,0] for k in range(200)])
				label_2 = np.array([[0,1] for k in range(200)])
				temp_train_y = np.row_stack((label_1,label_2))
			else:
				temp_train_y = np.row_stack(([0 for i in range(200)],[1 for i in range(200)]))
				temp_train_y = temp_train_y.flatten()
			
			temp_model = eval(model)(temp_train_x,temp_train_y)
			
			y_eval = temp_model.predict(temp_train_x)
			training_accuracy = metrics.accuracy_score(temp_train_y,y_eval)
			print('training accuracy: {}'.format("%.2f%%"%( 100*training_accuracy)))
			
			binary_trained_model[str(i)+'_'+str(j)] = copy.deepcopy(temp_model)
	return binary_trained_model
	
def binary_evaluation(model,test_data_x,test_data_y):
	
	count = 0
	predict = []
	for i in range(len(test_data_x)):
		label = [i for i in range(10)]
		temp_train = [test_data_x[i]]
		temp_label = test_data_y[i]
		while len(label)>1:
			num_1 = label[0]
			num_2 = label[-1]
			classifier = copy.deepcopy(model[str(num_1) + '_' + str(num_2)])
			ret = classifier.predict(temp_train)
			if ret.flatten()[0] == 0:
				label = label[:-1] # 删掉最后一个
			else:
				label = label[1:] # 删掉第一个
#			print(label)
			if len(label)==1:
				predict = copy.deepcopy(label[0])
			
		if int(temp_label)==int(predict):
			count += 1

	print("Accuracy: {}".format(count/len(test_data_x)))
			
	
if __name__ == '__main__':
	ret = binary_training('svm_classifier',one_hot=False)
	binary_evaluation(ret, x_test, y_test)
	
#	m = ret[str(1) + '_' + str(5)]
#	ret = m.predict([x_test[201]])
#	print(ret)