# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:14:55 2016

@author: ChrisChen
"""

import numpy as np
def normalEquation(theta, x, y):
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(x), x)), np.transpose(x)), y)
    
    return theta