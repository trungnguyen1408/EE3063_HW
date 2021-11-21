import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sgn(w, x):
  return np.sign(np.dot(w.T, x))

def perceptron(X, y, w_init, it=100):
  w = [w_init]
  i = 1
  while True: 
    pred =  sgn(w[-1], X).reshape(-1)
    pred[pred >= 0] = 1
    pred[pred < 0 ] = 0 

    # Get the id's point in convergence set
    mis_id = np.where(np.equal(pred, y) == False)[0]
    num_mis = mis_id.shape[0]
    if (num_mis == 0) or (i == it):
      return w, mis_id.size, i

    # Update weight
    random_id = np.random.choice(mis_id, 1)[0]
    w_t = w[-1] + (2 * y[random_id] - 1) * X[:, random_id].reshape(X_bar.shape[0], -1)   # size(w) = 3x1, size(X[:,0]) = (3,)
    w.append(w_t)

    i += 1


# Load data
path = './Iris.csv'
data = pd.read_csv(path)


# Iris-setosa data
X0 = data.iloc[0:50, 1:5].to_numpy().T
# Iris-virginica data
X1 = data.iloc[100:150, 1:5].to_numpy().T
X = np.concatenate((X0, X1), axis = 1)
# Xbar 
X_bar = np.concatenate((np.ones((1, X.shape[1])), X), axis = 0)
# Iris-virginica lable 1, Iris-setosa lable 0
y = np.array([0]*X0.shape[1] + [1]*X1.shape[1])


w_init = np.random.randn(X_bar.shape[0], 1)
w, mis_id, it = perceptron(X_bar, y, w_init)
print(w[-1].T)
print('Missing point: ', mis_id , ', it: ', it)