import numpy as np 
import matplotlib.pyplot as plt 

X = np.array([2, 6, 8, 4, 3, 1.5, 7, 1.7, 9, 6, 4, 2.3, 8.5, 7])
y = np.array([0, 1, 1, 0, 1,   0, 1, 0  , 1, 0, 1,   0, 1, 1])

X1 = X[y==0]
y1 = y[y==0]

print(X1)
print(y1)


plt.axis([0, 10, -1, 2])
plt.plot(X[y==0], y[y==0], "P")
plt.plot(X[y==1], y[y==1], "x")
plt.show()