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
        self.lossActivation = None

    def forward(self, X):
        """Complete a forward pass of the network for input X."""
        passForward = X
        for layer in self.layers:
            passForward = layer.forward(passForward, storeActivation=False)
        return passForward
        
    def _forward(self, X):
        """Complete a forward pass of the network for input X."""
        passForward = X
        for layer in self.layers:
            passForward = layer.forward(passForward, storeActivation=True)
        self.lossActivation = passForward
        return passForward

    def _backward(self, X, Y):
        """Backpropagate through the network for input X and output Y."""
        passBack = self.loss.backward(self.lossActivation,Y)
        WList = []
        bList = []
        for layer in reversed(self.layers):
            dW, db, passBack = layer.backward(passBack)
            WList.insert(0, dW)
            bList.insert(0, db)
        return WList, bList



