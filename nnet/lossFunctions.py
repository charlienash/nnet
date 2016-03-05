# -*- coding: utf-8 -*-
import numpy as np

class squaredError:
    
    def __init__(self):
        self.activation = None
        
    def loss(self, Y):
        return np.mean(np.sum(np.square(Y - self.activation), axis=1))

    def backward(self, Y):
        M = np.shape(Y)[0] # Batch-size
        return -2/M*(Y - self.activation)

