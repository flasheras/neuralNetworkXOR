# neuralNetworkXOR
Neural network to predict the outputs of XOr three param logic gates.

This example is taken from the book "Python All in one for Dummies" (https://www.amazon.es/Python-All-One-Dummies-Shovic/dp/1119557593). This book is a very good reference to start learning python. Takes you from not knowing anything about python (it was not my case) to implement a Neural Network (up to now). 

This repo is a personal aproach to Neural Networks, its internal behaviour and a simple example in python of how it works.

The file train.py launches the training while the neural_network class has the internal logic of a Neural Network.

Further work is needed as understand the algebra behind the scenes, understand the sigmoid functions, how weights impact in final result and how more or less layers and layer size impacts in the convergence or divergence of the model in the desired result.

The neural network implemented inferences the result of a XOR 3 params gate. 

# XOR behaviour example:

| Param 1 | Param 2 | Param 3 | Output |
|:---:|:---:|:---:|:---:|
|0|0|0|1|
|1|1|1|1|
|0|0|1|0|
|0|1|1|0|
|1|0|1|0|
|...|...|...|...|
