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
            passForward = layer.forward(passForward)
        return passForward
        
    def _forward(self, X):
        """Complete a forward pass of the network for input X storing the
        activations for backprop."""
        passForward = X
        for layer in self.layers:
            passForward = layer._forward(passForward, storeActivation=True)
        self.loss.activation = passForward
        return passForward

    def _backward(self, Y):
        """Backpropagate through the network for output Y."""
        passBack = self.loss._backward(Y)
        WList = []
        bList = []
        for layer in reversed(self.layers):
            dW, db, passBack = layer._backward(passBack)
            WList.insert(0, dW)
            bList.insert(0, db)
        return WList, bList
        
    def trainSGD(self, X, Y, learningRate=0.1, nEpochs=30, batchSize=100):
        
        nExamples = np.shape(X)[0]
        nBatches =  np.ceil(nExamples / batchSize).astype('int')
        for epoch in range(nEpochs):
            
            # Get current loss
            self._forward(X)
            loss = self.loss._loss(Y)
            print("Epoch: {}   Loss: {}".format(epoch+1, loss), flush=True)
            
            for b in range(nBatches):
                
                # Get minibatch
                miniBatchX = X[b*batchSize:(b+1)*batchSize,:]
                miniBatchY = Y[b*batchSize:(b+1)*batchSize,:]

                # Do minibatch update
                self._miniBatchUpdate(miniBatchX, miniBatchY, learningRate)
                   
                    
    def _miniBatchUpdate(self, minibatchX, minibatchY, learningRate=0.1):
        # Store activations with forward prop
        self._forward(minibatchX)
        
#        # Get current loss
#        loss = self.loss.loss(Y)
#        print("Current loss: {}".format(loss), flush=True)
        
        # Get gradients with backward prop
        WList, bList = self._backward(minibatchY)
        
        # Update params
        for gradW, gradB, layer in zip(WList, bList, self.layers):
            if layer.hasParams:
                layer.W = layer.W - learningRate*gradW 
                layer.b = layer.b - learningRate*gradB
                
#    def resetParams(self):
        
        
        



