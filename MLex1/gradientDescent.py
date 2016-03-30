# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 13:48:34 2016

@author: ChrisChen
"""
import numpy as np
# import computeCost

def gradientDescent(theta, iters, x, y, alpha):
    m = len(y)
    total = 0
    for i in range(iters):
        total = np.dot(np.transpose((np.dot(x, theta) - y)), x)
        theta = theta - np.transpose(alpha * total / m)
    
    return theta