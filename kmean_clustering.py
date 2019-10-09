import numpy as np 
from matplotlib import pyplot as plt 
from scipy.spatial.distance import cdist


means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
k = 3

X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0,X1,X2), axis=0)
labels = np.array([0]*N + [1]*N + [2]*N)


def init_center(X):
	return X[np.random.choice(X.shape[0], 3, replace=False), :]

def assign_label(X, center):
	D = cdist(X, center)
	return np.argmin(D, axis=1)

def assign_center(X, labels):
	centers = np.ones((3, 2))
	for i in range(centers.shape[0]):
		xk = X[labels==i, :]
		ck = np.mean(xk, axis=0)
		centers[i, :] = ck
	return centers

def has_converged(old_center, new_center):
	return (set([tuple(a) for a in old_center]) == 
        set([tuple(a) for a in new_center]))

def kmean_clustering(X, centers):
	old_centers = centers
	while True:
		labels = assign_label(X, old_centers)
		new_centers = assign_center(X, labels)
		if has_converged(old_centers, new_centers) is True:
			break
		old_centers = new_centers
	return new_centers, labels	

# first_centers = init_center(X)

# centers, labels = kmean_clustering(X, first_centers)

# print(centers)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
print('Centers found by scikit-learn:')
print(kmeans.cluster_centers_)
labels = kmeans.predict(X)

x0 = X[labels == 0, 0]
y0 = X[labels == 0, 1]
x1 = X[labels == 1, 0]
y1 = X[labels == 1, 1]
x2 = X[labels == 2, 0]
y2 = X[labels == 2, 1]



plt.plot(x0, y0, "s")
plt.plot(x1, y1, "1")
plt.plot(x2, y2, "P")
plt.show()
