# -*- coding: utf-8 -*-
import numpy as np

class squaredError:
    
    def __init__(self):
        self.activation = None
        
    def loss(self, Y, YPred):
        return np.mean(np.sum(np.square(Y - YPred), axis=1))
        
    def _loss(self, Y):
        return np.mean(np.sum(np.square(Y - self.activation), axis=1))

    def _backward(self, Y):
        M = np.shape(Y)[0] # Batch-size
        return -2/M*(Y - self.activation)
      
class logLoss:
    
    def __init__(self):
        self.activation = None
        
    def loss(self, Y, YPred):
        return -np.mean(np.log(YPred[Y.astype('bool')]))
        
    def _loss(self, Y):
        return -np.mean(np.log(self.activation[Y.astype('bool')]))

    def _backward(self, Y):
        M = np.shape(Y)[0] # Batch-size
        return -1/M * (Y / self.activation)
        

