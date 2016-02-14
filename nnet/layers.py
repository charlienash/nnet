import numpy as np
from .toolbox import sigmoid

class Layer:
    """Doc-string"""

    def __init__(self):
        pass

    def forward(self, input):
        raise NotImplementedError()

    def backward(self, grad):
        raise NotImplementedError()


class Linear(Layer):
    """Doc-string"""

    def __init__(self, inputDim, outputDim, rng):
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.rng = rng
        self.W = np.random.rand(outputDim, inputDim)
        self.b = np.random.rand(outputDim)
        self.activation = np.empty(inputDim)

    def forward(self, X, storeActivation=False):
        if storeActivation:           
            self.activation = X
        return self.W.dot(X) + self.b

    def backward(self, backGrad):
        dW = np.outer(backGrad, self.activation)
        db = backGrad
        inputGrad = self.W.T.dot(backGrad)
        return dW, db, inputGrad
        
class Sigmoid(Layer):
    """Doc-string"""
    
    def __init__(self):
        self.activation = None

    def forward(self, X):
        self.activation = X
        return sigmoid(X)

    def backward(self, backGrad):
        X = self.activation
        dW = None
        db = None
        inputGrad = backGrad*sigmoid(X)*(1 - sigmoid(X)) # Element-wise product
        return dW, db, inputGrad
        
class ReLu(Layer):
    """Doc-string"""
    
    def __init__(self):
        self.activation = None

    def forward(self, X):
        self.activation = X
        return (X + np.abs(X))/2

    def backward(self, backGrad):
        X = self.activation
        dW = None
        db = None
        inputGrad = backGrad*(0.5 + np.abs(X)*0.5) # Element-wise product
        return dW, db, inputGrad
    
    
