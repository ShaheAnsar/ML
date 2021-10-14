import numpy as np
from os import path as p
from struct import *
from matplotlib import pyplot as plt


class MNIST_loader:
    def __init__(self, basepath="./datasets/mnist/", **kwargs):
        paths = [
            "train-images-idx3-ubyte",
            "train-labels-idx1-ubyte",
            "t10k-images-idx3-ubyte",
            "t10k-labels-idx1-ubyte",
            ]
        paths = [p.join(basepath, i) for i in paths]
        self.train_set = []
        self.test_set = []

        for i in range(2):
            data = None
            with open(paths[i], "rb") as f:
                data = f.read()
            d = int.from_bytes(data[0:4], signed=False, byteorder="big")
            if d == 2049:
                self.train_set.append(np.frombuffer(data[8:], dtype='B'))
            elif d == 2051:
                imgs = np.frombuffer(data[16:], dtype='B')
                imgs = np.reshape(imgs, (60000, 28, 28))
                self.train_set.append(imgs)
                
        for i in range(2,4):
            data = None
            with open(paths[i], "rb") as f:
                data = f.read()
            d = int.from_bytes(data[0:4], signed=False, byteorder="big")
            if d == 2049:
                self.test_set.append(np.frombuffer(data[8:], dtype='B'))
            elif d == 2051:
                imgs = np.frombuffer(data[16:], dtype='B')
                imgs = np.reshape(imgs, (10000, 28, 28))
                self.test_set.append(imgs)
