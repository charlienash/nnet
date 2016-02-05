import numpy as np

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

    def forward(self, X):
        return self.W.dot(X) + self.b

    def backward(self, backGrad, X):
        dW = backGrad.dot(X)
        db = backGrad
        inputGrad = self.W.dot(backGrad)
        return dW, db, inputGrad
