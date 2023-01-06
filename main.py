import numpy as np
import cv2 as cv
import json
import os
from keras.datasets import mnist

np.set_printoptions(suppress=True)


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.outputs = []
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.outputs = SoftMax(np.dot(inputs, self.weights) + self.biases)


def SoftMax(inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return probabilities


def toPredicted(result):
    a = np.zeros(10)
    b = a.tolist()
    b[result] = 1

    return b


def ImageToPixel(path):
    img = cv.imread(path)
    pixels = []
    for x in img:
        for y in x:
            pixels.append(np.sum(np.divide(y, 765)))
    return pixels


class Network:
    def __init__(self, n_inputs, n_layers, n_neurons, n_outputs, name):
        self.predicted = []
        self.outputs = []
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.Layers = []
        self.inputs = []
        self.Layers.append(Layer_Dense(n_inputs, n_neurons))
        for i in range(1, n_layers - 1):
            self.Layers.append(Layer_Dense(n_neurons, n_neurons))
        self.Layers.append(Layer_Dense(n_neurons, n_outputs))
        self.name = name
        if os.path.isfile(f"{name}weights.json") and os.path.isfile(f"{name}biases.json"):
            self.Load()
            print("data was loaded successfully")

    def forward(self, inputs):
        inputs = np.ndarray.flatten(inputs)
        self.Layers[0].forward(inputs)
        for i in range(1, self.n_layers):
            self.Layers[i].forward(self.Layers[i - 1].outputs)
        self.outputs = self.Layers[self.n_layers - 1].outputs

    def error(self, layer):
        if layer == self.n_layers - 1:
            r = np.multiply(SoftMax(self.outputs), np.subtract(self.outputs, self.predicted))
        else:
            r = np.multiply(SoftMax(self.Layers[layer].outputs),
                            np.sum(np.multiply(self.error(layer + 1), self.Layers[layer].outputs)))
        return np.mean(r)

    def WeightDelta(self, layer):
        if layer == 0:
            return np.multiply(self.inputs, self.error(layer))
        else:
            return np.multiply(self.Layers[layer - 1].outputs, self.error(layer))

    def BiasDelta(self, layer):
        return self.error(layer)

    def Save(self):
        all_weights = np.asarray([layer.weights for layer in self.Layers])
        all_biases = np.asarray([layer.biases for layer in self.Layers], dtype=object)

        np.savez(f"{self.name}weights", weights=all_weights, biases=all_biases)

    def Load(self):
        datafile = np.load(f"{self.name}weights.npz")
        raw_weights = datafile["weights"]
        all_weights = np.asarray(raw_weights)
        raw_biases = datafile["biases"]
        all_biases = np.asarray(raw_biases)
        for layer in range(self.n_layers):
            self.Layers[layer].weights = all_weights[layer]
            self.Layers[layer].biases = all_biases[layer]

    def train(self, inputs, predicted, learning_rate):
        self.inputs = inputs
        self.predicted = predicted
        for i in range(10000):
            self.forward(inputs)
            for layer in range(self.n_layers):
                if layer == self.n_layers - 1:
                    self.Layers[layer].weights = np.subtract(self.Layers[layer].weights,
                                                             np.multiply(self.WeightDelta(layer), learning_rate).T)

                else:
                    self.Layers[layer].weights = np.subtract(self.Layers[layer].weights,
                                                             np.multiply(self.WeightDelta(layer), learning_rate).T)
                self.Layers[layer].biases = np.subtract(self.Layers[layer].biases,
                                                        np.multiply(self.BiasDelta(layer), learning_rate))

        # self.Save()


nn = Network(784, 3, 12, 10, "first")
(train_X, train_y), (test_X, test_y) = mnist.load_data()
last = 0
for index in range(len(train_X)):
    nn.train(np.reshape(train_X[index], (1, 784)), toPredicted(train_y[index]), -0.1)
    print(index)


right = 0
wrong = 0
for index in range(len(test_X)):
    nn.forward(np.ndarray.flatten(test_X[index]))
    if np.argmax(nn.outputs) == test_y[index]:
        right += 1
    else:
        wrong += 1
print("rights percentage:")
print(right * 100 / len(test_X))
print("rights count:")
print(right)
