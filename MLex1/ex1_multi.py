# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:45:09 2016

@author: ChrisChen
"""

import numpy as np
import matplotlib.pyplot as plt
import featureNormalize as fn
import gradientDescent as gd
import normalEqn as ne

house = np.loadtxt('ex1data2.txt', delimiter = ',')
x = house[:, 0:2]
y = house[:, 2]
y = y.reshape(47, 1)

x_fn = fn.featureNormalize(x)
m = len(x)
x_intercept = np.ones((m, 1))
x_total = np.concatenate((x_intercept, x_fn), axis = 1)
thetas1 = np.zeros((x_total.shape[1], 1))

iterations = 400
alpha = 0.03

# using the gradientDescent function from univariate linear regression
thetas1 = gd.gradientDescent(thetas1, iterations, x_total, y, alpha)

# normal equations
x = house[:, 0:2]
y = house[:, 2]
y = y.reshape(47, 1)
x_intercept = np.ones((m, 1))
x_total = np.concatenate((x_intercept, x), axis = 1)
thetas2 = np.zeros((x_total.shape[1], 1))
thetas2 = ne.normalEquation(thetas2, x_total, y)
