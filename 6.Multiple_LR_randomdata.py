# Multiple Linear Regression using Random Data

import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def generate_dataset(n):
    x = []
    y = []
    for i in range(1, n + 1):
        x1 = i
        x2 = i / 2 + np.random.rand() * n
        x.append([1, x1, x2])
        y.append(x1 * x1 + x2 * x2 + 1)
    print("Generated X values ", np.array(x))
    print("Generated Y values ", np.array(y))
    return np.array(x), np.array(y)


def mse(coef, x, y):
    return np.mean((np.dot(x, coef) - y) ** 2) / 2


def gradients(coef, x, y):
    return np.mean(x.transpose() * (np.dot(x, coef) - y), axis=1)


def multilinear_regression(coef, x, y, lr, b1=0.9, b2=0.999, epsilon=1e-8):
    prev_error = 0
    m_coef = np.zeros(coef.shape)
    v_coef = np.zeros(coef.shape)
    moment_m_coef = np.zeros(coef.shape)
    moment_v_coef = np.zeros(coef.shape)
    t = 0
    while True:
        error = mse(coef, x, y)
        if abs(error - prev_error) <= epsilon:
            break
        prev_error = error
        grad = gradients(coef, x, y)
        t += 1
        m_coef = b1 * m_coef + (1 - b1) * grad
        v_coef = b2 * v_coef + (1 - b2) * grad ** 2
        moment_m_coef = m_coef / (1 - b1 ** t)
        moment_v_coef = v_coef / (1 - b2 ** t)
        delta = ((lr / moment_v_coef ** 0.5 + 1e-8) *
                 (b1 * moment_m_coef + (1 - b1) * grad / (1 - b1 ** t)))
        coef = np.subtract(coef, delta)
    return coef


# driver code
x, y = generate_dataset(50)

mpl.rcParams['legend.fontsize'] = 12

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.scatter(x[:, 1], x[:, 2], y, label='y values', s=5)
ax.legend(loc='best')
ax.view_init(45, 0)

plt.show()

coef = np.array([0, 0, 0])
c = multilinear_regression(coef, x, y, 1e-1)
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.scatter(x[:, 1], x[:, 2], y, label='y values',
           s=5, color='dodgerblue')
ax.scatter(x[:,1], x[:, 2], c[0] + c[1] * x[:, 1] + c[2] * x[:, 2],
           label='Regression values', s=5, color="orange")

ax.view_init(45, 0)
ax.legend(loc='best')
plt.show()
