import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np


# Truth table. All possible values. Entry of the neural network.
X = np.array(([0, 0, 0], [0, 0, 1], [0, 1, 0],
              [0, 1, 1], [1, 0, 0], [1, 0, 1],
              [1, 1, 0], [1, 1, 1]), dtype=float)  # 7x3 Tensor

# Suppervised method: the result of our prediction
y = np.array(([1], [0], [0], [0], [0],
             [0], [0], [1]), dtype=float)

# Define the layers and activation functions
model = tf.keras.Sequential()
model.add(Dense(4, input_dim=3, activation='relu', use_bias=True))
model.add(Dense(4, activation='relu', use_bias=True))
model.add(Dense(1, activation='sigmoid', use_bias=True))

# Compile the method
model.compile(loss='mean_squared_error', optimizer='adam',
              metrics=['binary_accuracy'])

# Print initial weights
print(model.get_weights())

# Fit the method
history = model.fit(X, y, epochs=2000, validation_data=(X, y))

# Print the summary.
model.summary()

# Print out to file
loss_history = history.history["loss"]
numpy_loss_history = np.array(loss_history)
np.savetxt("loss_history.txt", numpy_loss_history, delimiter="\n")
binary_accuracy_history = history.history["binary_accuracy"]
numpy_binary_accuracy = np.array(binary_accuracy_history)
np.savetxt("binary_accuracy.txt", numpy_binary_accuracy, delimiter="\n")
print(np.mean(history.history["binary_accuracy"]))

# Test the model
result = model.predict(X).round()
print(result)
