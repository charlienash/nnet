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
        """Complete a forward pass of the network for input minibatch X. 
        
        X is a NxD numpy array."""
        passForward = X
        for layer in self.layers:
            passForward = layer.forward(passForward, storeActivation=False)
        return passForward
        
    def _forward(self, X):
        """Complete a forward pass of the network for input X storing the
        activations for backprop."""
        passForward = X
        for layer in self.layers:
            passForward = layer.forward(passForward, storeActivation=True)
        self.loss.activation = passForward
        return passForward

    def _backward(self, Y):
        """Backpropagate through the network for output Y."""
        passBack = self.loss.backward(Y)
        WList = []
        bList = []
        for layer in reversed(self.layers):
            dW, db, passBack = layer.backward(passBack)
            WList.insert(0, dW)
            bList.insert(0, db)
        return WList, bList
        
    def optimize(self, X, Y, stepSize=0.1, nIters=100):
        
        for i in range(nIters):

            # Store activations with forward prop
            self._forward(X)
            
            # Get current loss
            loss = self.loss.loss(Y)
            print("Current loss: {}".format(loss), flush=True)
            
            # Get gradients with backward prop
            WList, bList = self._backward(Y)
            
            # Update params
            for gradW, gradB, layer in zip(WList, bList, self.layers):
                if layer.hasParams:
                    layer.W = layer.W - stepSize*gradW 
                    layer.b = layer.b - stepSize*gradB
        
        



