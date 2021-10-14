import numpy as np
import json
class NeuralNetwork:
    def __init__(self, layers, learn_rate = 0.1):
        self.W = []
        self.layers = layers
        self.lr = learn_rate
        for i in range(len(layers) - 2):
            W = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            W /= np.sqrt(layers[i])
            self.W.append(W)

        W = np.random.randn(layers[-2] + 1, layers[-1])
        W /= np.sqrt(layers[-2])
        self.W.append(W)

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return np.exp(-x)/(1 + np.exp(-x))**2

    def fit(self, X, y, epochs=1000, display_update = 100):
        X = np.c_[ X, np.ones((X.shape[0])) ]

        for epoch in range(epochs):
            for (x, target) in zip(X, y):
                self.partial_fit(x, target)

            if epoch % display_update == 0:
                print(f"Loss = {self.calculate_loss(X, y)}")

    def partial_fit(self, x, target):

        A = [x]
        nets = []
        # Feedforward stage
        for i in range(len(self.W)):
            net = A[-1].dot(self.W[i])
            a = self.sigmoid(net)
            nets.append(net)
            A.append(a)

        # Backpropagation stage
        error = A[-1] - target

        # Multiply each of these with the transpose of the activations
        # to get the partial derivative matrix of L with respect to W_i
        grad_prefixes = [self.sigmoid_prime(nets[-1])*error]

        for i in range(2,  len(A)):
            # The partial derivative vector of the loss functon
            # with respect to the activations of the i-1th layer
            l_f = self.W[-i + 1].dot(grad_prefixes[-1].T)
            grad_prefix = l_f * self.sigmoid_prime(nets[-i])
            grad_prefixes.append(grad_prefix)

        grad_prefixes = grad_prefixes[::-1]

        for i in range(len(self.W)):
            self.W[i] += -self.lr * grad_prefixes[i]*np.c_[A[i]]

    def predict(self, X, add_bias=True):
        p = np.atleast_2d(X)
        if add_bias:
            p = np.c_[X, np.ones((X.shape[0]))]

        for i in range(len(self.W)):
            p = self.sigmoid(p.dot(self.W[i]))

        return p

    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, add_bias=False)
        return 0.5*np.sum((targets - predictions)**2)

    def dump(self, filename):
        with open(filename, "w") as f:
            print(json.dumps([w.tolist() for w in self.W]))
            f.write(json.dumps([w.tolist() for w in self.W]))
