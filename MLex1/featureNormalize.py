# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:46:26 2016

@author: ChrisChen
"""

import numpy as np

def featureNormalize(x):
    mu = np.zeros((1, x.shape[1]))
    sigma = np.zeros((1, x.shape[1]))
    for i in range(x.shape[1]):
        mu[0, i] = x[:, i].mean()
        sigma[0, i] = x[:, i].std()
        
    for j in range(x.shape[0]):
        x[j, :] = (x[j, :] - mu) / sigma
    
    return x
    