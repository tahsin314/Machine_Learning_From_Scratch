import numpy as np
import matplotlib.pyplot as plt


class linearRegression:

    def __init__(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.X = X
        self.y = y.reshape((len(y),1))
        self.X_dim = list(self.X.shape)
        self.y_dim = list(self.y.shape)
        self.m = len(self.X)
        self.W = np.random.randn(self.X.shape[1],1)
        self.Cost_all = [] 
        
    def linearCost(self):
        h = np.dot(self.X, self.W)
        loss = h - self.y
        cost = np.sum(loss**2) / (2*self.m)
        return loss, cost

    def gradientDescent(self, learningRate, numIterations):
        for i in range(numIterations):
            self.numIterations = numIterations
            loss, cost = self.linearCost()
            self.Cost_all.append(cost)
            print("Iteration %d || Cost: %f" % (i+1, cost))
            gradient = np.dot(np.transpose(self.X), loss) / self.m
            self.W -= learningRate * gradient
        return self.W    

    def plotCost(self):
        plt.plot([i for i in range(self.numIterations)],self.Cost_all)
        plt.xlabel("Number of iterations")
        plt.ylabel("Cost")
        plt.show()


""" Creating data with weight value 80.0 and bias value 30.0"""
R1 = np.random.random(1000)
R2 = np.random.random(1000)
X = [[1.0, r1, r2] for r1, r2 in zip(R1, R2)]  # 1.0 for bias term
y = [(80.0*r1+30.0*r2 + 15.0) for r1, r2 in zip(R1, R2)]

""" Training..."""
linearRegression = linearRegression(X, y)
print(linearRegression.gradientDescent(learningRate=0.02, numIterations=5000))
linearRegression.plotCost()
