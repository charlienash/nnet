import nnet
import numpy as np

rng = np.random.RandomState()

layers = [
          nnet.Linear(5,3, rng),
          nnet.Linear(3,2, rng)
          ]

loss = nnet.lossFunctions.squaredError()

neuralNet = nnet.NeuralNetwork(layers, loss, rng)

neuralNet.forward(np.ones(5))
neuralNet.backward(np.ones(5), np.array([1,5]))