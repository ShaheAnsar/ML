from neuralnet import NeuralNetwork
import numpy as np

nn = NeuralNetwork([2, 2, 1])
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]
X = np.array(X)
y = np.array(y)
nn.fit(X, y, epochs=10000)
print(nn.predict(X))
