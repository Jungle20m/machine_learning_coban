import numpy as np 
from matplotlib import pyplot as plt


X = np.array([[147, 153, 155, 158, 160, 163, 165, 168, 173, 175, 178, 180, 183, 150]])
y = np.array([[49, 51, 52, 54, 56, 58, 59, 60, 63, 64, 66, 67, 68, 50]]).T
one = np.ones([1, 14])

Xbar = np.concatenate((one, X), axis=0).T

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)

w = np.dot(np.linalg.pinv(A), b)

print(w.shape)

x0 = np.array([140, 190])
y0 = w[1][0]*x0 + w[0][0]


plt.plot(x0, y0)
plt.plot(X.T, y, "P")
plt.show()

print(w)

def loss(w):
	return 0.5*np.linalg.norm(y-Xbar.dot(w))**2

print(loss(w))

