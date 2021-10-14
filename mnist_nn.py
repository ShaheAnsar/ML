from neuralnet import NeuralNetwork
from mnist_loader import MNIST_loader
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import cv2

loader = MNIST_loader()

train_set = [np.zeros((loader.train_set[0].shape[0], 64)),
             LabelBinarizer().fit_transform(loader.train_set[1])]
test_set = [np.zeros((loader.test_set[0].shape[0], 64)),
            LabelBinarizer().fit_transform( loader.test_set[1] )]

for i in range(loader.train_set[0].shape[0]):
    train_set[0][i] = cv2.resize(loader.train_set[0][i], (8,8)).flatten()/255.0
for i in range(loader.test_set[0].shape[0]):
    test_set[0][i] = cv2.resize(loader.test_set[0][i], (8,8)).flatten()/255.0


nn = NeuralNetwork([64, 32, 16, 10], learn_rate=0.5)
train_len = 2000
print(test_set[1][:20])
nn.fit(train_set[0][:train_len], train_set[1][:train_len], epochs=5000)
test_len = 100
p = nn.predict(test_set[0][:test_len])
print(classification_report(test_set[1][:test_len].argmax(axis = 1), p.argmax(axis = 1)))
