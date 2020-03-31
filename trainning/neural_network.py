import numpy as np


class Neural_Network(object):
    lossFile = open("SumSquaredLossList.csv", "w")
    """2 Layer neural network with default values"""
    def __init__(self, input_layer_size=3, output_layer_size=1,
                 hidden_layer_size=4):
        # Attributes
        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size
        self.hidden_layer_size = hidden_layer_size
        """ Weight between input and hidden layers. Matrix of (0,1]
        3 is the dimension of the input data for one value.
        This is a 3x4 matrix from input to hidden
        X (N x 3)  dot [w1,w1,w1,w1 = (N x4)
                        w1,w1,w1,w1
                        w1,w1,w1,w1]
        """
        self.W1 = np.random.rand(self.input_layer_size, self.hidden_layer_size)
        """ Weight between hidden and output layers. Matrix of (0,1]
        4x1 Matrix. Nx4 dot 4 x 1 = N x 1 (The desired result)
        [h1,h1 ,h1, h1 dot [w2 = [o1
        h1,h1 ,h1, h1       w2    o2
        h1,h1 ,h1, h1]      w2    03
                            w2]   04]
        """
        self.W2 = np.random.rand(self.hidden_layer_size,
                                 self.output_layer_size)

    def feedForward(self, X):
        # feedForward propagation through our network
        # dot product of X (input) and first set of 3x4 weights
        self.z = np.dot(X, self.W1)
        # the activationSigmoid activation function - neural magic
        self.z2 = self.activationSigmoid(self.z)
        # dot product of hidden layer (z2) and second set of 4x1 weights
        self.z3 = np.dot(self.z2, self.W2)
        # final activation function - more neural magic
        o = self.activationSigmoid(self.z3)
        return o

    def backwardPropagate(self, X, y, o):
        # backward propagate through the network
        # calculate the error in output
        self.o_error = y - o
        # apply derivative of activationSigmoid to error
        self.o_delta = self.o_error*self.activationSigmoidPrime(o)
        # z2 error: how much our hidden layer weights contributed to output
        # error
        self.z2_error = self.o_delta.dot(self.W2.T)
        # applying derivative of activationSigmoid to z2 error
        self.z2_delta = self.z2_error*self.activationSigmoidPrime(self.z2)
        # adjusting first set (inputLayer --> hiddenLayer) weights
        self.W1 += X.T.dot(self.z2_delta)
        # adjusting second set (hiddenLayer --> outputLayer)weights
        self.W2 += self.z2.T.dot(self.o_delta)

    def trainNetwork(self, X, y):
        # feed forward the loop
        o = self.feedForward(X)
        # and then back propagate the values (feedback)
        self.backwardPropagate(X, y, o)

    def activationSigmoid(self, s):
        # activation function
        # simple activationSigmoid curve as in the book
        return 1/(1+np.exp(-s))

    def activationSigmoidPrime(self, s):
        # First derivative of activationSigmoid # calculus time!
        return s * (1 - s)

    def saveSumSquaredLossList(self, i, error):
        self.lossFile.write(str(i) + "," + str(error.tolist())+'\n')

    def saveWeights(self):
        # save this in order to reproduce our cool nwtwork
        np.savetxt("weightsLayer1.txt", self.W1, fmt="%s")
        np.savetxt("weightsLayer2.txt", self.W2, fmt="%s")

    def predictOutput(self, xPredicted):
        print("Predicted XOR output data based on trained weights: ")
        print("Expected (X1-X3): \n" + str(xPredicted))
        print("Output (Y1): \n" + str(self.feedForward(xPredicted)))
