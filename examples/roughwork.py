import nnet
import nnet.layers as netLayers
import numpy as np
np.set_printoptions(precision=4)

rng = np.random.RandomState()

layers = [
          nnet.Linear(5, 10, rng),
          netLayers.ReLu(),  
          nnet.Linear(10, 2, rng)
          ]

loss = nnet.lossFunctions.squaredError()
neuralNet = nnet.NeuralNetwork(layers, loss, rng)

X = np.random.rand(10,5)
#Y = np.random.rand(10,2)
W = np.random.rand(2,5)
Y = X.dot(W.T) + np.random.normal(0, 0.1, [10,2])
#np.sum(np.square(Y - YPred), axis=1)
#neuralNet._forward(X)
#WList, bList = neuralNet._backward(Y)

neuralNet.optimize(X,Y, nIters=10000, stepSize=0.5)
YPred = neuralNet.forward(X)
