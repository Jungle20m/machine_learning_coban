import numpy as np 
from matplotlib import pyplot as plt 


X = np.array([[147, 153, 155, 158, 160, 163, 165, 168, 173, 175, 178, 180, 183, 150]])
y = np.array([[49, 51, 52, 54, 56, 58, 59, 60, 63, 64, 66, 67, 68, 50]]).T
one = np.ones([1, 14])

Xbar = np.concatenate((one, X), axis=0).T

def grad(w):
	return Xbar.T.dot(Xbar.dot(w)-y)

def loss(w):
	return 0.5/14*np.linalg.norm(y-Xbar.dot(w))**2

w = np.array([[-34.25801910],[0.56324466]])

print(w)

print(grad(w))

print(np.linalg.norm(grad(w)))

# print("#"*30)
# w1 = w - 0.2*grad(w)

# print(w1)

# print(grad(w1))

# print(np.linalg.norm(grad(w1)))

# print("#"*30)
# w2 = w1 - 0.2*grad(w1)

# print(w2)

# print(grad(w2))

# print(np.linalg.norm(grad(w2)))

a = -34.25802789 - (-8.78000006e-06)
b =  0.56179046 - (-1.45420001e-03)

print(a)
print(b)





