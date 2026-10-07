import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
# from csvloader import MNIST_Example
from csvloader import *
# from matplotlib import pyplot as plt
# from keras.datasets import mnist
# from tensorflow.keras.datasets import mnist
# from mnist import MNIST

# mndata = MNIST('./python-mnist')
# images, labels = mndata.load_training()



NUM_LAYERS = 3
NN_LAYERS = [
	28 * 28,
	10,
	10
]







class gradientVectorClass:
	def __init__(self, weightArrayList, biasArrayList):
		self.weights = weightArrayList
		self.biases = biasArrayList

	def copy(self):
		return gradientVectorClass([
				self.weights[0].copy(),
				self.weights[1].copy(),
				self.weights[2].copy()
			],
			[
				self.biases[0].copy(),
				self.biases[1].copy(),
				self.biases[2].copy()
			])

	def takeAverage(self, num):
		self.weights[0] /= num
		self.weights[1] /= num
		self.weights[2] /= num
		self.biases[0] /= num
		self.biases[1] /= num
		self.biases[2] /= num

	def negativeGradient(self):
		self.weights[0] *= -1
		self.weights[1] *= -1
		self.weights[2] *= -1
		self.biases[0] *= -1
		self.biases[1] *= -1
		self.biases[2] *= -1

	def toArray(self):
		return [
			[
				self.weights[0].copy(),
				self.weights[1].copy(),
				self.weights[2].copy()
			],
			[
				self.biases[0].copy(),
				self.biases[1].copy(),
				self.biases[2].copy()
			]
		]





# ----- Weight initialization -----
def xavierInit(fan_out, fan_in):
	limit = np.sqrt(6. / (fan_in + fan_out))
	return np.random.uniform(low=-limit, high=limit, size=(fan_out, fan_in))

def heInit(fan_out, fan_in):
	stddev = np.sqrt(2. / fan_in)
	return np.random.normal(loc=0., scale=stddev, size=(fan_out, fan_in))





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
	shift_arr = arr - np.max(arr, axis=0, keepdims=True)
	exp_arr = np.exp(shift_arr)
	exp_sum = np.sum(exp_arr, axis=0, keepdims=True)
	return exp_arr / exp_sum

def softmax1D(arr):
	shift_arr = arr - np.max(arr)
	exp_arr = np.exp(shift_arr)
	exp_sum = np.sum(exp_arr)
	return exp_arr / exp_sum

def derivativeSoftmax(z): # Derivative of softmax(z_j) wrt z_j
	a = softmax1D(z)
	return a * (1 - a)





# A 784 x 128 x 10 neural network for MNIST classification
class NeuralNetwork:
	def __init__(self, actFunc, derivativeActFunc, wInit, learningRate, hiddenSize=128):
		self.actFunc = actFunc
		self.dActFunc = derivativeActFunc
		self.hiddenSize = hiddenSize

		self.neuronLayers = [
			np.zeros((28 * 28, BATCH_SIZE)),
			np.zeros((self.hiddenSize, BATCH_SIZE)),
			np.zeros((10, BATCH_SIZE))
		]

		# 3D array - layers - right neuron index - left neuron index
		self.weights = [
			np.zeros((1, 1)), # First layer is useless
			np.zeros((self.hiddenSize, 28 * 28)),
			np.zeros((10, self.hiddenSize))
		]

		weightInitScale = 1.0
		wInit = wInit.upper()
		if wInit == "NAIVE":
			# ----- Naive random initialization
			self.weights[1] = np.random.rand(self.hiddenSize, 28 * 28) * weightInitScale
			self.weights[2] = np.random.rand(10, self.hiddenSize) * weightInitScale
		elif wInit == "XAVIER":
			# ----- Xavier initialization (Uniform dist)
			self.weights[1] = xavierInit(self.hiddenSize, 28 * 28)
			self.weights[2] = xavierInit(10, self.hiddenSize)
		elif wInit == "HE":
			# ----- He initialization (Normal dist)
			self.weights[1] = heInit(self.hiddenSize, 28 * 28)
			self.weights[2] = heInit(10, self.hiddenSize)

		biasInitScale = 0.0
		self.bias = [
			None,
			np.full((self.hiddenSize, BATCH_SIZE), biasInitScale),
			np.full((10, BATCH_SIZE), biasInitScale)
		]

		self.zLinear = [
			None,
			np.zeros((self.hiddenSize, BATCH_SIZE)),
			np.zeros((10, BATCH_SIZE))
		]

		self.learningRate = learningRate
		self.accuracies = []
		self.losses = []

	# ----- By 3B1B -----
	def getPredictions(self):
		return np.argmax(self.neuronLayers[2], axis=0)

	def getAccuracy(self, predictions, labels):
		print(f"pred: {predictions}, labels: {labels}")
		accuracy = np.sum(predictions == labels) / labels.size
		self.accuracies.append(accuracy)
		return accuracy

	def getLoss(self, desiredOutput):
		# Categorical cross-entropy loss over batch: -sum(y * log(a + eps)) / BATCH_SIZE
		batchLoss = -np.sum(desiredOutput * np.log(self.neuronLayers[2] + 1e-15)) / BATCH_SIZE
		self.losses.append(batchLoss)
		return batchLoss

	def forwardPropagation(self, exampleBatch):
		exampleImages = exampleBatch.images
		print(f"labels: {exampleBatch.labels}")
		# print(example.image)

		self.neuronLayers[0][:, :] = exampleImages.copy()

		self.zLinear[1][:, :] = np.matmul(self.weights[1], self.neuronLayers[0]) + self.bias[1]
		# self.neuronLayers[1][:, :] = ReLU(self.zLinear[1][:, :])
		self.neuronLayers[1][:, :] = self.actFunc(self.zLinear[1][:, :])

		self.zLinear[2][:, :] = np.matmul(self.weights[2], self.neuronLayers[1]) + self.bias[2]
		self.neuronLayers[2][:, :] = softmax(self.zLinear[2][:, :])

		# print(f"output: {self.neuronLayers[2]}")      # --- Print the output neurons
		# print(np.sum(self.neuronLayers[2], axis=0))   # --- Make sure all neurons sum to 1
		batchAccuracy = self.getAccuracy(self.getPredictions(), exampleBatch.labels)
		batchLoss = self.getLoss(exampleBatch.desiredOutputs())
		print(f"Acc. of batch: {batchAccuracy:.4f}, Loss: {batchLoss:.4f}")   # --- Print accuracy and loss of the batch
		return self.neuronLayers[2].copy()

	# INSPIRED BY 3B1B
	def calculateGradientBatch(self, exampleBatch):
		gradientVector = gradientVectorClass(
			[
				np.zeros((1, 1)), # First layer is useless
				np.zeros((self.hiddenSize, 28 * 28)),
				np.zeros((10, self.hiddenSize))
			], # Weights
			[
				np.zeros(1), # First layer is useless
				np.zeros(self.hiddenSize),
				np.zeros(10)
			] # Biases
		)

		desiredOutput = exampleBatch.desiredOutputs()

		# For Categorical Cross-Entropy with Softmax:
		# dL / dz2 = a2 - y
		delta2 = self.neuronLayers[2] - desiredOutput

		# Loop over all examples in batch
		for i in range(BATCH_SIZE):
			delta2_i = delta2[:, i]

			# --- Backpropagate layer 1 (calculate derivative with respect to zLinear of layer 1)
			# delta1 = (W2^T @ delta2) * dActFunc(z1)
			delta1_i = np.matmul(self.weights[2].transpose(), delta2_i) * self.dActFunc(self.zLinear[1][:, i])

			# --- Accumulate derivatives with respect to WEIGHTS
			gradientVector.weights[2][:, :] += np.outer(delta2_i, self.neuronLayers[1][:, i])
			gradientVector.weights[1][:, :] += np.outer(delta1_i, self.neuronLayers[0][:, i])

			# --- Accumulate derivatives with respect to BIASES
			gradientVector.biases[2][:] += delta2_i
			gradientVector.biases[1][:] += delta1_i

		gradientVector.takeAverage(BATCH_SIZE)

		return gradientVector

	def gradientDescent(self, gradientVector):
		negGradientVector = gradientVector.copy()
		# negGradientVector.negativeGradient()

		self.weights[1] -= negGradientVector.weights[1] * self.learningRate
		self.weights[2] -= negGradientVector.weights[2] * self.learningRate

		self.bias[1] -= np.tile(negGradientVector.biases[1], (BATCH_SIZE, 1)).transpose() * self.learningRate
		self.bias[2] -= np.tile(negGradientVector.biases[2], (BATCH_SIZE, 1)).transpose() * self.learningRate

	def overallAccuracy(self):
		return sum(self.accuracies) / len(self.accuracies) if self.accuracies else 0.0

	def overallLoss(self):
		return sum(self.losses) / len(self.losses) if self.losses else 0.0


	# ----- Pandas write weights and biases to CSV file -----
	def writeWeightsToCSV(self, folderName):
		weightDataFrame = pd.DataFrame(data=self.weights[1],
			index=list(range(self.weights[1].shape[0])),
			columns=list(range(self.weights[1].shape[1])))
		weightDataFrame.to_csv(f"{folderName}/weights/layer1.csv", encoding="utf-8")
		weightDataFrame = pd.DataFrame(data=self.weights[2],
			index=list(range(self.weights[2].shape[0])),
			columns=list(range(self.weights[2].shape[1])))
		weightDataFrame.to_csv(f"{folderName}/weights/layer2.csv", encoding="utf-8")
		print("Written weights successfully!")

	def writeBiasToCSV(self, folderName):
		biasDataFrame1 = pd.DataFrame(data=self.bias[1][:, 0], columns=["layer1"])
		biasDataFrame1.to_csv(f"{folderName}/bias/layer1.csv", encoding="utf-8")
		biasDataFrame2 = pd.DataFrame(data=self.bias[2][:, 0], columns=["layer2"])
		biasDataFrame2.to_csv(f"{folderName}/bias/layer2.csv", encoding="utf-8")
		print("Written biases successfully!")


	# ----- Pandas read weights and biases from CSV file -----
	def readFromCSV(self, folderName):
		import os
		weights1 = np.array(pd.read_csv(f"{folderName}/weights/layer1.csv"))
		weights2 = np.array(pd.read_csv(f"{folderName}/weights/layer2.csv"))

		self.weights[1] = weights1[:, 1:]
		self.weights[2] = weights2[:, 1:]

		# Adapt hidden size to match the loaded weights
		self.hiddenSize = self.weights[1].shape[0]
		self.neuronLayers[1] = np.zeros((self.hiddenSize, BATCH_SIZE))
		self.zLinear[1] = np.zeros((self.hiddenSize, BATCH_SIZE))

		if os.path.exists(f"{folderName}/bias/layer1.csv") and os.path.exists(f"{folderName}/bias/layer2.csv"):
			bias1 = np.array(pd.read_csv(f"{folderName}/bias/layer1.csv"))[:, 1]
			bias2 = np.array(pd.read_csv(f"{folderName}/bias/layer2.csv"))[:, 1]
			self.bias[1] = np.tile(bias1, (BATCH_SIZE, 1)).transpose()
			self.bias[2] = np.tile(bias2, (BATCH_SIZE, 1)).transpose()
		elif os.path.exists(f"{folderName}/bias/bias.csv"):
			bias = np.array(pd.read_csv(f"{folderName}/bias/bias.csv"))
			self.bias[1] = np.tile(bias[:, 1], (BATCH_SIZE, 1)).transpose()
			self.bias[2] = np.tile(bias[:, 2], (BATCH_SIZE, 1)).transpose()


	# ----- Forward propagation (not for training or testing) -----
	def forwardPropagationNormal(self, exampleBatch):
		exampleImages = exampleBatch.images
		# print(f"labels: {exampleBatch.labels}")
		# print(example.image)

		self.neuronLayers[0][:, :] = exampleImages.copy()

		self.zLinear[1][:, :] = np.matmul(self.weights[1], self.neuronLayers[0]) + self.bias[1]
		# self.neuronLayers[1][:, :] = ReLU(self.zLinear[1][:, :])
		self.neuronLayers[1][:, :] = self.actFunc(self.zLinear[1][:, :])

		self.zLinear[2][:, :] = np.matmul(self.weights[2], self.neuronLayers[1]) + self.bias[2]
		self.neuronLayers[2][:, :] = softmax(self.zLinear[2][:, :])

		# print(f"output: {self.neuronLayers[2]}")      # --- Print the output neurons
		# print(np.sum(self.neuronLayers[2], axis=0))   # --- Make sure all neurons sum to 1
		# print(self.getAccuracy(self.getPredictions(), exampleBatch.labels)) # --- Print accuracy
		return self.neuronLayers[2].copy()

