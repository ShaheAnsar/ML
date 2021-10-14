import numpy as np

class Perceptron:
    def __init__(self, N, learn_rate = 0.1):
        self.W = np.random.randn(N + 1)/np.sqrt(N)
        self.lr = learn_rate

    def stepn(self, x):
        r = x
        r[r > 0] = 1.0
        r[r < 0] = 0.0
        return r
    def step(self, x):
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs = 10):
        X = np.c_[X, np.ones((X.shape[0]))]

        for epoch in range(epochs):
            for (x, target) in zip(X, y):
                p = self.step(x.dot(self.W))
                if p != target:
                    delta = target - p
                    self.W += x * delta * self.lr

    def predict(self, X, addBias=True):
        X = np.atleast_2d(X)
        if addBias:
            X = np.c_[X, np.ones((X.shape[0]))]
        return self.stepn(np.dot(X, self.W))

X = np.array([[0, 0], [0,1], [1, 0], [1, 1]]) # OR dataset
y = [0, 1, 1, 0]
p = Perceptron(X.shape[1])
print("Training!")
p.fit(X, y, epochs = 100)
print("Trained")
print("Predicting")
pred = p.predict(X)
for (p, t) in zip(pred, y):
    print(f"Predicted: {p}, Target: {t}")
