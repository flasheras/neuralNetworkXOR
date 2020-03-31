import numpy as np
from neural_network import Neural_Network

# X = input of our 3 input XOR gate
# set up the inputs of the neural network (right from the table)
# We are going to train the neural network with X & y.
X = np.array(([0, 0, 0], [0, 0, 1], [0, 1, 0],
              [0, 1, 1], [1, 0, 0], [1, 0, 1],
              [1, 1, 0], [1, 1, 1]), dtype=float)  # 7x3 Tensor
# y = our output of our neural network. This is a supervised method.
y = np.array(([1], [0], [0], [0], [0],
             [0], [0], [1]), dtype=float)
# what value we want to predict
xPredicted = np.array(([0, 0, 1]), dtype=float)

# Normalize xPredicted
X = X/np.amax(X, axis=0)  # maximum of X input array
# maximum of xPredicted (our input data for the prediction)
xPredicted = xPredicted/np.amax(xPredicted, axis=0)

# set up our Loss file for graphing
lossFile = open("SumSquaredLossList.csv", "w")

myNeuralNetwork = Neural_Network(hidden_layer_size=10)
# trainingEpochs = 1000
trainingEpochs = 100000
for i in range(trainingEpochs):
    # train myNeuralNetwork 1,000 times print ("Epoch # " + str(i) + "\n")
    print("Network Input : \n" + str(X))
    print("Expected Output of XOR Gate Neural Network: \n" + str(y))
    print("Actual Output from XOR Gate Neural Network: \n" +
          str(myNeuralNetwork.feedForward(X)))  # mean sum squared loss
    Loss = np.mean(np.square(y - myNeuralNetwork.feedForward(X)))
    myNeuralNetwork.saveSumSquaredLossList(i, Loss)
    print("Sum Squared Loss: \n" + str(Loss))
    print("\n")
    myNeuralNetwork.trainNetwork(X, y)

myNeuralNetwork.saveWeights()
myNeuralNetwork.predictOutput(xPredicted)
