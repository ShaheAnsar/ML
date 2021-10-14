#! /usr/bin/env python3
import cv2
from mnist_loader import MNIST_loader
from pprint import pprint


import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from math import floor

def eucl_dist(img1, img2):
    return np.sum(np.sqrt( np.square(img1 - img2) ))

def manh_dist(img1, img2):
    return np.sum(np.abs(img1 - img2))

class KNN_Classifier:
    def __init__(self, train_set):
        self.train_set = train_set
        self.enum_train_imgs = [(i, img) for i, img in enumerate(self.train_set[0])]

    def predict(self, test_data, k):
        labels = np.zeros(10)
        dists = [(eucl_dist(test_data, img), i) for i, img in self.enum_train_imgs]
        dists.sort(key=lambda x: x[0])
        #plt.subplot(1, k + 1, 1)
        #plt.imshow(test_data)
        #for i in range(k):
        #    plt.subplot(1, k + 1, 2 + i)
        #    plt.imshow(self.train_set[0][dists[i][1]])
        #plt.show()
        for i in range(k):
            labels[self.train_set[1][ dists[i][1] ]] += 1
        return max(enumerate(labels), key= lambda x: x[1])[0]
    def score(self, test_set, k=5):
        correct_count = 0
        total_count = 0
        for i, img in enumerate(test_set[0]):
            label = self.predict(img,k)
            if label == test_set[1][i]:
                correct_count += 1
            total_count += 1
            if total_count % 100 == 0:
                print(f"Current success rate {100.0 * correct_count/total_count}")

        print(f"This classifier has a success rate of {100.0 * correct_count/total_count}")
loader = MNIST_loader()
train_set = [np.zeros(( loader.train_set[0].shape[0], 8,8 )), loader.train_set[1]]
test_set = [np.zeros(( loader.test_set[0].shape[0], 8,8 )), loader.test_set[1]]
#train_set = loader.train_set
#test_set = loader.test_set
for i in range(loader.train_set[0].shape[0]):
    train_set[0][i] = cv2.resize(loader.train_set[0][i], (8,8))
for i in range(loader.test_set[0].shape[0]):
    test_set[0][i] = cv2.resize(loader.test_set[0][i], (8,8))
for i in range(10):
    plt.subplot(5,4, i + 1)
    plt.imshow(train_set[0][i])
for i in range(10):
    plt.subplot(5,4, 11 + i)
    plt.imshow(train_set[0][10 + i])
plt.show()
mem_len = 50000
classifier = KNN_Classifier([train_set[0][:mem_len], train_set[1][:mem_len]])
classifier.score([test_set[0][:floor(mem_len/10)], test_set[1][:floor(mem_len/10)]], k = 3)
