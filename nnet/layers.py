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
        self.W = np.random.rand([outputDim, inputDim])
        self.b = np.random.rand(outputDim)

    def forward(self, input):
        return self.W.dot(input) + self.b

    def backward(self, grad, X):
        return
