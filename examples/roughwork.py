import nnet
import nnet.layers as layers
import numpy as np
np.set_printoptions(precision=3)

rng = np.random.RandomState()

layers = [
          layers.Linear(5, 10, rng),
          layers.ReLu(),  
          layers.Linear(10, 3, rng),
          layers.SoftMax()
          ]
loss = nnet.lossFunctions.logLoss()
neuralNet = nnet.NeuralNetwork(layers, loss, rng)

nExamples = 10000
X = np.random.randn(nExamples, 5)
#Y = np.random.rand(10,2)
W = np.random.randn(3,5)
#Y = X.dot(W.T) + np.random.normal(0, 0.1, [nExamples,2])
Y = np.zeros([nExamples, 3])
for i,row in enumerate(Y):
#    ID = np.random.randint(0, 2)
#    row[ID] = 1
#    row[0] = 1
    x = X[i]    
    row[np.argmax(W.dot(x))] = 1
Y = Y.astype('int')

#np.sum(np.square(Y - YPred), axis=1)
#neuralNet._forward(X)
#WList, bList = neuralNet._backward(Y)

trainParams = {
                'nEpochs' : 100,
                'batchSize' : 100,
                'learningRate' : 0.5
              }
neuralNet.trainSGD(X,Y, **trainParams)
YPred = neuralNet.forward(X)
hardPredict = np.zeros([nExamples, 3])
for i, row in enumerate(hardPredict):
    softPred = YPred[i]
    row[np.argmax(softPred)] = 1
hardPredict = hardPredict.astype('int')
acc = np.sum((Y == hardPredict).any(axis=1)) / nExamples

#YPred = np.random.rand(100,10)
#YPred = YPred / np.sum(YPred, axis=1)[:,np.newaxis]
#YGrad = np.random.rand(nExamples, 5)

#def softmax(X):
#    return np.exp(X) / np.sum(np.exp(X), axis=1)[:,np.newaxis]
    
# Jacobian of softmax
#Y = np.array([[1,0,0,0,0]])
#X = np.random.randn(1,5)
#YPred = softmax(X)
#M = np.shape(Y)[0] # Batch-size
#YGrad = -1/M * (Y / YPred)
#Jac = -YPred[:,:,np.newaxis]*YPred[:,np.newaxis,:]
#di = np.diag_indices(5,2)
#Jac[:,di[0], di[1]] = Jac[:,di[0], di[1]] + YPred
#XGrad = np.sum(Jac*YGrad[:,np.newaxis,:], axis=2)
#YPred - Y




