import numpy as np

class NeuralNetwork:
    """
    Doc-string
    """

    def __init__(self, layers, loss, rng=None):
        self.layers = layers
        self.loss = loss
        if rng is None:
            print("No random state specified. Initialising random state.")
            self.rng = np.random.RandomState()

    def forward(self, X):
        """Complete a forward pass of the network for input X."""
        passForward = X
        for layer in self.layers:
            passForward = layer.forward(passForward)
        return passForward

    def backward(self, X, Y):
        """Backpropagate through the network for input X and output Y."""
        passBack = self.loss.backward(X,Y)
        gradList = []
        for layer in reversed(self.layers):
            passBack = layer.backward(passBack)
            gradList.append(passBack)
        return gradList



