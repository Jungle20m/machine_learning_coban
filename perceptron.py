import numpy as np 
from matplotlib import pyplot as plt 


means = [[1, 4], [4, 1]]
conv = [[1, 0], [0, 1]]
N =8

X1 = np.random.multivariate_normal(means[0], conv, N)
X2 = np.random.multivariate_normal(means[1], conv, N)
X = np.concatenate((X1, X2), axis=0)
one = np.ones((N*2, 1))

X = np.concatenate((one, X), axis=1)
labels = np.array([1]*N + [-1]*N).T

def init_w():
	return np.array([1, 1, 1])

def loss_func(J):
	loss = 0
	for j in J:
		if j > 0:
			loss += j
	return loss

def perceptron(X, labels, eta):
	w = init_w()
	for k in range(40):
		J = -labels*X.dot(w)
		if loss_func(J) == 0:
			return w
		for i in range(2*N):
			print(i)
			w = w + eta*labels[i]*X[i,:]

w = perceptron(X, labels, 1)

# w = init_w()

# w = w + labels[0]*X[0, :]
print(w)
# print(-labels*X.dot(w))


if w[2] != 0:
	x = np.array((0, 10))
	y = -(w[1]/w[2])*x - w[0]/w[2]
else:
	x = [-w[0]/w[1], -w[0]/w[1]]
	y = [0, 10]


print(y)

plt.plot(X1[:, 0], X1[:, 1], "P")
plt.plot(X2[:, 0], X2[:, 1], "x")
plt.plot(x, y)
plt.show()