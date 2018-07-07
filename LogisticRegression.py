"""This is my code for logistic regression. You can use the class below and
 call functions. Just remember: Your labels should be
in hot vector form.
I used bangla digit data for testing. You can use any other data
you want.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm


class logisticRegression():

    def __init__(self, X_train, y_train, X_test, y_test):
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_tmp = np.ones((X_train.shape[0], X_train.shape[1]+1))
        X_tmp[:, 1:] = X_train
        X_train = X_tmp
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        self.X_train = X_train
        X_tmp = np.ones((X_test.shape[0], X_test.shape[1] + 1))
        X_tmp[:, 1:] = X_test
        self.X_test = X_tmp
        try:
            self.y_train = y_train.reshape((len(y_train), y_train.shape[1]))
            self.y_test = y_test.reshape((len(y_test), y_test.shape[1]))
            self.W = 1e-2*np.random.randn(self.X_train.shape[1], y_train.shape[1])
        except:
            self.y_train = y_train.reshape((len(y_train), 1))
            self.y_test = y_test.reshape((len(y_test), 1))
            self.W = 1e-2*np.random.randn(self.X_train.shape[1], 1)
        self.X_dim = list(self.X_train.shape)
        self.y_dim = list(self.y_train.shape)
        self.m = len(self.X_train)
        self.Cost_all = []
        self.Acc_all = []

    def sig(self, x, w):
        return 1 / (1 + np.exp(-np.dot(x, w)))
        
    def logisticCost(self, alpha):
        self.sigmoid = self.sig(self.X_train, self.W)
        loss = -(np.multiply(self.y_train, np.log(self.sigmoid)) + np.multiply((1-self.y_train), np.log(1-self.sigmoid)))
        cost = np.sum(loss) / len(self.X_train)
        cost += (alpha/(2*len(self.X_train)))*np.sum(np.multiply(self.W[1:], self.W[1:]))
        return loss, cost

    def prediction_singleclass(self):
        pred = self.sig(self.X_test, self.W)
        Predictions = []
        for p in pred:
            if p < 0.5:
                Predictions.append(0.0)
            else:
                Predictions.append(1.0)
        return np.sum(y_test == np.array(Predictions)) / len(X_test)

    def prediction_multiclass(self):
        pred = self.sig(self.X_test, self.W)
        correct = 0.0
        for p in range(len(pred)):
            if(np.argmax(pred[p]) == np.argmax(self.y_test[p])):
                correct += 1.0
        return correct/len(pred)

    def gradientDescent(self, learningRate, numIterations, multiclass= True, batch_num = 1, reg = 0):
        self.numIterations = numIterations
        batch_size = int(len(self.X_train)*1.0/batch_num)
        for i in tqdm(range(numIterations)):
            for b in range(batch_num):
                self.X_train_batch = self.X_train[b*batch_size:(b+1)*batch_size]
                self.y_train_batch = self.y_train[b*batch_size:(b+1)*batch_size]
                loss, cost = self.logisticCost(reg)
                gradient = np.dot(np.transpose(self.X_train), self.sigmoid - self.y_train) / batch_size
                self.W -= learningRate * gradient

            self.X_train_batch = self.X_train[batch_num*batch_size:]
            self.y_train_batch = self.y_train[batch_num*batch_size:]
            loss, cost = self.logisticCost(reg)
            gradient = np.dot(np.transpose(self.X_train), self.sigmoid - self.y_train) / batch_size
            gradient += (reg/len(self.X_train))*np.sum(self.W[1:])
            self.W -= learningRate * gradient

            if i % 10 == 0:
                if multiclass == True:
                    print("Iteration %d || Cost: %f || Acc %f" % (i+1, cost, self.prediction_multiclass()))
                else:
                    print("Iteration %d || Cost: %f || Acc %f " % (i+1, cost, self.prediction_singleclass()))
                if learningRate >= 1e-5:
                    learningRate *= 0.999
            acc = self.prediction_multiclass()
            self.Cost_all.append(cost) # Uncomment this line and call plotCost() for plotting cost
            self.Acc_all.append(acc) # Uncomment this line and call plotAcc() for plotting accuracy
        print("Final accuracy: ", self.prediction_multiclass())
        print("Best accuracy: ", np.max(self.Acc_all))
        return self.W

    def plotCost(self):
        plt.plot([i for i in range(self.numIterations)], self.Cost_all)
        plt.xlabel("Number of iterations")
        plt.ylabel("Cost")
        plt.show()

    def plotAcc(self):
        plt.plot([i for i in range(self.numIterations)], self.Acc_all)
        plt.xlabel("Number of iterations")
        plt.ylabel("Accuracy")
        plt.show()


""" Data pre-processing """
from sklearn import preprocessing
from scipy.io import loadmat
bangla = loadmat('./mldata/ISI_BN_Trn18000Tst4000(XBN).mat')
X_train, y_train, X_test, y_test = bangla['train_x'], bangla['train_y'], bangla['test_x'], bangla['test_y']
X_train = preprocessing.normalize(X_train, norm='max', axis=1)
X_test = preprocessing.normalize(X_test, norm='max', axis=1)

"""If your label data is not in hot vector form, uncomment the lines below to convert."""
# hot_vec = np.zeros((len(y_train), int(np.max(y_train))+1))
# for i in range(len(hot_vec)):
#     hot_vec[i,int(y_train[i])] = 1
#y_train = hot_vec

# hot_vec = np.zeros((len(y_test), int(np.max(y_test))+1))
# for i in range(len(hot_vec)):
#     hot_vec[i,int(y_test[i])] = 1
#y_test = hot_vec

""" Feeding and applying Logistic Regression """
logisticRegression = logisticRegression(X_train, y_train, X_test, y_test)
logisticRegression.gradientDescent(learningRate=1e-2, numIterations=1000, batch_num=100, reg=0.5)
logisticRegression.plotCost()
logisticRegression.plotAcc()
