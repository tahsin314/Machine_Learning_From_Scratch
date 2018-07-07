
import numpy as np
from sklearn import datasets


class SVM:
	"""This is actually a poor implementation of Support Vector Machine inspired
	by Siraj Raval. I didn't add hard margin or soft margin option to the code.
	Also this is only linear SVM. Maybe later sometime I will add polynomial
	SVM and RBF SVM. """

	def __init__(self, X_train, y_train, X_test, y_test):
		X_tmp = np.ones((X_train.shape[0], X_train.shape[1]+1))
		X_tmp[:,1:] = X_train
		self.X_train = X_tmp
		X_tmp = np.ones((X_test.shape[0], X_test.shape[1]+1))
		X_tmp[:,1:] = X_test
		self.X_test = X_tmp
		self.W = np.zeros((X_train.shape[1]+1))
		self.y_train = y_train
		self.y_test = y_test

	def Linearprediction(self):
		pred = np.dot(self.X_test, self.W.T)
		return pred

	def Linearaccuracy(self, y):
		pred = self.Linearprediction()
		for p in range(len(pred)):
			if pred[p] > 1:
				pred[p] = 1
			else:
				pred[p] = -1
		return 1.0*np.sum(pred == y)/len(y)

	def LinearSVM(self, learning_rate, num_iteration, reg=0):
		best_acc = 0.0
		for step in range(num_iteration):
			er = 0
			for i in range(len(self.X_train)):
				if (np.dot(self.y_train[i],np.dot(self.X_train[i],self.W.T)))<=0:
					self.W += learning_rate*(self.X_train[i]*self.y_train[i] - 2*reg*self.W) *(.99)**step
					er +=1.0/len(self.X_train)
				else:
					self.W += -learning_rate*2*reg*self.W
			accu = self.Linearaccuracy(y_test)
			print("Iteration: ",step, "|| Accuracy: ",accu, "|| Cost:", er)
			if accu> best_acc:
				best_acc = accu
				best_W = self.W
		return best_acc, best_W

from sklearn.utils import shuffle
from sklearn import preprocessing
breast = datasets.load_breast_cancer()
X_, y_ = breast.data[:, :], breast.target[:]

"""Data pre-processing"""
X, y = shuffle(X_, y_)
X_train, y_train = X[:500], y[:500]
X_test, y_test = X[500:], y[500:]
X_train, X_test = preprocessing.normalize(X_train, norm='max'), preprocessing.normalize(X_test, norm='max')

for i in range(len(y_train)):
	if y_train[i] == 0:
		y_train[i] = -1

for i in range(len(y_test)):
	if y_test[i] == 0:
		y_test[i] =- 1

svm = SVM(X_train, y_train, X_test, y_test)
b_acc, b_w = svm.LinearSVM(learning_rate=10, num_iteration=10000, reg=0)
# print("Best:", b_acc, b_w)

