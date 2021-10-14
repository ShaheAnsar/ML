from keras.models import Sequential
from keras.layers.core import Dense
from tensorflow.keras.optimizers import SGD
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import numpy as np

dataset = datasets.fetch_openml("mnist_784")
data = dataset.data.astype("float")/255.0
(trainX, testX, trainY, testY) = train_test_split(data, dataset.target, test_size=0.25)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation="softmax"))
sgd = SGD(0.1)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs = 20, batch_size=128)
p = model.predict(testX)
print(classification_report(testY.argmax(1),
                            p.argmax(1),
                            target_names = [str(x) for x in lb.classes_]))
for k,_ in H.history:
    print(k)
plt.style.use("ggplot")
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.legend()
plt.show()
