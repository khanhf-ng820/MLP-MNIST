# More general neural network class

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt



NAIVE = "NAIVE"
XAVIER = "XAVIER"
HE = "HE"


# WEIGHT INITIALIZATION
def naiveInit(weights):
	shape = weights.shape
	newWeights = np.random.rand(*shape)
	weights[:, :] = newWeights

def xavierInit(weights):
	shape = weights.shape
	fan_out, fan_in = shape
	limit = np.sqrt(6. / (fan_in + fan_out))
	newWeights = np.random.uniform(low=-limit, high=limit, size=shape)
	weights[:, :] = newWeights

def heInit(weights):
	shape = weights.shape
	fan_out, fan_in = shape
	stddev = np.sqrt(2. / fan_in)
	newWeights = np.random.normal(loc=0., scale=stddev, size=shape)
	weights[:, :] = newWeights



# ----- Activation functions and derivatives -----
def ReLU(z):
	return np.maximum(0, z)

def derivativeReLU(z):
	return np.heaviside(z, 0)

def leakyReLU(z):
	return np.maximum(0.1 * z, z)

def derivativeLeakyReLU(z):
	return np.heaviside(z, 0) * 0.9 + 0.1

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def derivativeSigmoid(z):
	sig = sigmoid(z)
	return sig * (1 - sig)

def softmax(arr):
	exp_arr = np.exp(arr)
	# print(f"arr: {arr}")
	# print(f"sumexp: {np.sum(exp_arr, axis=0)}")
	exp_sum = np.tile(np.sum(exp_arr, axis=0), (np.size(exp_arr, 0), 1))
	exp_arr /= exp_sum
	return exp_arr

def softmax1D(arr):
	exp_arr = np.exp(arr)
	exp_sum = np.sum(exp_arr)
	return exp_arr / exp_sum

def derivativeSoftmax(z): # Derivative of softmax(z_j) wrt z_j
	a = softmax1D(z)
	return a * (1 - a)



# WILL COME BACK LATER
class Layer:
	def __init__(self, batchSize, numNeurons, actFunc=None, derivActFunc=None, prevLayer=None, wInit=None):
		self.batchSize = batchSize
		self.actFunc = actFunc
		self.dActFunc = derivActFunc

		self.neurons = np.zeros((numNeurons, batchSize))
		self.bias = np.zeros(numNeurons)
		self.z = np.zeros((numNeurons, batchSize))
		if prevLayer is None:
			self.weights = None
			self.prevNeurons = None
		else:
			numPrevNeurons = prevLayer
			self.weights = np.zeros((numNeurons, numPrevNeurons))
			self.prevNeurons = prevLayer.neurons.view()

		# Initialize weights
		if wInit == NAIVE:
			naiveInit(self.weights.view())
		elif wInit == XAVIER:
			xavierInit(self.weights.view())
		elif wInit == HE:
			heInit(self.weights.view())

	def calculateA(self):
		if self.weights is None:
			return
		self.z = np.matmul(self.weights, self.prevNeurons) + np.tile(self.bias, (self.batchSize, 1)).T
		self.neurons = self.actFunc(self.z)



class NeuralNetwork:
	def __init__(self, batchSize, neuronsArr, afuncArr, dAfuncArr, wInit=NAIVE):
		self.batchSize = batchSize
		self.numLayers = len(neuronsArr)
		self.neurons = [Layer(self.batchSize, neuronsArr[0])]

		for i in range(1, self.numLayers):
			self.neurons.append(Layer(
				self.batchSize, neuronsArr[i], afuncArr[i], dAfuncArr[i], self.neurons[-1], wInit
			))

	def forwardPropagation(self, exampleBatch):
		exampleImages = exampleBatch.images
		print(f"labels: {exampleBatch.labels}")
		# print(example.image)

		self.neuronLayers[0][:, :] = exampleImages.copy()

		# self.zLinear[1][:, :] = np.matmul(self.weights[1], self.neuronLayers[0]) + self.bias[1]
		# self.neuronLayers[1][:, :] = self.actFunc(self.zLinear[1][:, :])

		# self.zLinear[2][:, :] = np.matmul(self.weights[2], self.neuronLayers[1]) + self.bias[2]
		# self.neuronLayers[2][:, :] = softmax(self.zLinear[2][:, :])
		for i in range(1, self.numLayers):
			self.neurons[i].calculateA()

		# # print(f"output: {self.neuronLayers[2]}")      # --- Print the output neurons
		# # print(np.sum(self.neuronLayers[2], axis=0))   # --- Make sure all neurons sum to 1
		# print(f"Acc. of batch: {self.getAccuracy(self.getPredictions(), exampleBatch.labels)}") # --- Print accuracy
		# return self.neuronLayers[2].copy()

