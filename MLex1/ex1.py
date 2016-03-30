import numpy as np
import matplotlib.pyplot as plt
import computeCost
import gradientDescent

foodtruck = np.loadtxt('ex1data1.txt', delimiter = ',')

plt.plot(foodtruck[:, 0], foodtruck[:, 1], '^')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
#plt.show()

x = foodtruck[:, 0]
y = foodtruck[:, 1]
x = x.reshape(97, 1)
y = y.reshape(97, 1)
m = len(x)
x_intercept = np.ones((m, 1))
x_total = np.concatenate((x_intercept, x), axis = 1)
thetas = np.zeros((2, 1))
iterations = 1500
alpha = 0.01

J = computeCost.computeCost(x_total, y, thetas)
print(J)

thetas = gradientDescent.gradientDescent(thetas, iterations, x_total, y, alpha)
print(thetas)

plt.plot(x_total[:, 1], np.dot(x_total, thetas), '-')