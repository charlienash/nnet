import nnet
import nnet.layers as netLayers
import numpy as np
np.set_printoptions(precision=4)

rng = np.random.RandomState()

layers = [
          nnet.Linear(5, 3, rng),
          netLayers.ReLu(),  
          nnet.Linear(3, 2, rng)
          ]

loss = nnet.lossFunctions.squaredError()

neuralNet = nnet.NeuralNetwork(layers, loss, rng)

neuralNet.forward(np.ones(5))
WList, bList = neuralNet.backward(np.ones(5), np.array([1,5]))