import numpy as np

def sigmoid(x):
  return 1/(np.exp(-x)+1)

def grad_sigmoid(x):
  return sigmoid(x)*(1-sigmoid(x))

x = np.array([0.1, 0.2])

W1 = np.array([0.1, 0.3, 0.4, 0.2, 0.15, 0.6]).reshape(3,2)

W2 = np.array([0.4, 0.25, 0.15, 0.4, 0.2, 0.3, 0.1, 0.25, 0.15]).reshape(3,3)

W3 = np.array([0.55, 0.1, 0.25, 0.45, 0.7, 0.4]).reshape(2,3)

b = np.array([0.6, 0.6, 0.6])


z_h = W1.dot(x) + b[0]
h = sigmoid(z_h)

z_k = W2.dot(h) + b[1]
k = sigmoid(z_k)

z_o = W3.dot(k) + b[2]
o = sigmoid(z_o)

print('z_h: ', z_h)
print('h: ', h)
print('z_k: ', z_k)
print('k: ', k)
print('z_o: ', z_o)
print('o: ', o)

E = 0.5 * ((1.0 - o[0]) ** 2 + (0 - o[1]) ** 2)
print(E)

y = np.array([1, 0])

dE_dzo1 = -(y[0] - o[0]) * grad_sigmoid(z_o[0])
dE_dzo2 = -(y[1] - o[1]) * grad_sigmoid(z_o[1])

print('dE_dzo1: ', dE_dzo1)
print('dE_dzo2: ', dE_dzo2)

dE_dzk1 = (dE_dzo1 * W3[0][0] + dE_dzo2 * W3[1][0]) * grad_sigmoid(z_k[0])
dE_dzk2 = (dE_dzo1 * W3[0][1] + dE_dzo2 * W3[1][1]) * grad_sigmoid(z_k[1])
dE_dzk3 = (dE_dzo1 * W3[0][2] + dE_dzo2 * W3[1][2]) * grad_sigmoid(z_k[2])

print('dE_dzk1: ', dE_dzk1)
print('dE_dzk2: ', dE_dzk2)
print('dE_dzk1: ', dE_dzk3)

dE_dzh1 = (dE_dzk1 * W2[0][0] + dE_dzk2 * W2[1][0] + dE_dzk3 * W2[2][0]) * grad_sigmoid(z_h[0])
dE_dzh2 = (dE_dzk1 * W2[0][1] + dE_dzk2 * W2[1][1] + dE_dzk3 * W2[2][1]) * grad_sigmoid(z_h[1])
dE_dzh3 = (dE_dzk1 * W2[0][2] + dE_dzk2 * W2[1][2] + dE_dzk3 * W2[2][2]) * grad_sigmoid(z_h[2])

print('dE_dzh1: ', dE_dzh1)
print('dE_dzh2: ', dE_dzh2)
print('dE_dzh3: ', dE_dzh3)