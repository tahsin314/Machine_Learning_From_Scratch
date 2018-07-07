"""This code visualize K means- clustered data with three classes. I used iris data
 from sklearn.load dataset. There are some bugs associated with this. Most of them
are due to bugs in matplotlib."""
from sklearn import datasets
import random 
import numpy as np 
from matplotlib import pyplot as plt

K = np.zeros(3)
iris = datasets.load_iris()
X, y = iris.data[:, :3], iris.target[:]

plt.scatter(X[:50, 0], X[:50, 1], marker='s', color='red')
plt.scatter(X[50:100, 0], X[50:100, 1], marker='o', color='blue')
plt.scatter(X[100:, 0], X[100:, 1], marker='v', color='green')
plt.savefig('Actual.jpg')
plt.show()

K = random.sample(([i for i in range(X.shape[0])]), len(K))
K = [X[i] for i in K]
print(K)
C = [0 for i in range(len(X))]
d = [0 for i in range(len(K))]

for t in range(25):
	for i in range(len(X)):
		for k in range(len(K)):
			d[k] = np.linalg.norm(X[i] - K[k])
		C[i] = d.index(min(d))
	X0, X1, X2 = [],[],[]
	# print(C)
	for i in range(len(C)):
		if C[i] == 0:
			X0.append(X[i])
		elif C[i] == 1:
			X1.append(X[i])
		else:
			X2.append(X[i])
	X0 = np.array(X0)
	X1 = np.array(X1)
	X2 = np.array(X2)
	K[0] = np.array([np.mean(X0[:,0]), np.mean(X0[:,1]), np.mean(X0[:,2])])
	K[1] = np.array([np.mean(X1[:,0]), np.mean(X1[:,1]), np.mean(X1[:,2])])
	K[2] = np.array([np.mean(X2[:,0]), np.mean(X2[:,1]), np.mean(X2[:,2])])

plt.scatter(X0[:, 0], X0[:, 1], marker='s', color='red')
plt.scatter(X1[:, 0], X1[:, 1], marker='o', color='blue')
plt.scatter(X2[:, 0], X2[:, 1], marker='v', color='green')
plt.savefig("Final"+'.jpg')
plt.show()

