import numpy as np
def computeCost(x, y, theta):
    m = len(y)
    total = 0
    
    total = np.sum(np.square(np.dot(x, theta) - y))
    J = total / (2 * m)
    return J