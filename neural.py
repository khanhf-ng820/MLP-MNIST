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





# A 784 x 10 x 10 neural network for MNIST classification
class NeuralNetwork:
	def __init__(self, actFunc, derivativeActFunc, wInit, learningRate):
		self.actFunc = actFunc
		self.dActFunc = derivativeActFunc

		self.neuronLayers = [] # 3D array - layers - neuron index - example
		self.neuronLayers.append(np.random.rand(28*28, BATCH_SIZE))
		self.neuronLayers.append(np.random.rand(10, BATCH_SIZE))
		self.neuronLayers.append(np.random.rand(10, BATCH_SIZE))

		self.weights = [] # 3D array - layers - right neuron index - left neuron index
		weightInitScale = 1.0
		self.weights.append(np.random.rand(1, 1) * weightInitScale) # First layer is useless
		self.weights.append(np.random.rand(10, 28*28) * weightInitScale)
		self.weights.append(np.random.rand(10, 10   ) * weightInitScale)

		wInit = wInit.upper()
		if wInit == "NAIVE":
			# ----- Naive random initialization
			self.weights[1] = np.random.rand(10, 28*28) * weightInitScale
			self.weights[2] = np.random.rand(10, 10   ) * weightInitScale
		elif wInit == "XAVIER":
			# ----- Xavier initialization (Normal dist)
			self.weights[1] = xavierInit(10, 28*28)
			self.weights[2] = xavierInit(10, 10   )
		elif wInit == "HE":
			# ----- He initialization (Normal dist)
			self.weights[1] = heInit(10, 28*28)
			self.weights[2] = heInit(10, 10   )

		biasInitScale = 0.
		# 3D array - layers - neuron index - example
		# First layer is useless
		self.bias = np.full((3, 10, BATCH_SIZE), biasInitScale)

		# 3D array - layers - neuron index - example
		# First layer is useless
		self.zLinear = np.zeros((3, 10, BATCH_SIZE)) # Linear combinations of each layer

		self.learningRate = learningRate
		self.accuracies = []

	# ----- By 3B1B -----
	def getPredictions(self):
		return np.argmax(self.neuronLayers[2], axis=0)

	def getAccuracy(self, predictions, labels):
		print(f"pred: {predictions}, labels: {labels}")
		accuracy = np.sum(predictions == labels) / labels.size
		self.accuracies.append(accuracy)
		return accuracy

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
		print(f"Acc. of batch: {self.getAccuracy(self.getPredictions(), exampleBatch.labels)}") # --- Print accuracy
		return self.neuronLayers[2].copy()

	# INSPIRED BY 3B1B
	def calculateGradientBatch(self, exampleBatch):
		gradientVector = gradientVectorClass(
			[
				np.zeros((1,1)), # First layer is useless
				np.zeros((10, 28*28)),
				np.zeros((10, 10))
			], # Weights
			[
				np.zeros(1), # First layer is useless
				np.zeros(10),
				np.zeros(10)
			] # Biases
		) # Weights first, biases later
		# Derivative of C_i (loss function) with respect to weights and biases, respectively
		# Take average over all examples

		derivativeActivation = [
			np.zeros((28*28, BATCH_SIZE)),
			np.zeros((10, BATCH_SIZE)),
			np.zeros((10, BATCH_SIZE))
		] # Derivative of C_i (loss function) with respect to activation values
		# Take average over all examples

		# Apply loss function (cost function)
		desiredOutput = exampleBatch.desiredOutputs()
		lossOfOutput = np.square(desiredOutput - self.neuronLayers[2])

		# --- Backpropagate layer 2 (calculate derivative with respect to ACTIVATION VALUES)
		derivativeActivation[2][:, :] = 2 * (self.neuronLayers[2] - desiredOutput)

		# Loop over all examples
		for i in range(BATCH_SIZE):
			# --- Backpropagate layer 1 (calculate derivative with respect to ACTIVATION VALUES)
			# k is before j
			derivativeActivation[1][:, i] = np.matmul(
				self.weights[2].transpose(),
				derivativeSoftmax(self.zLinear[2][:, i]) * derivativeActivation[2][:, i]
			)

			# --- Backpropagate layer 0 (calculate derivative with respect to ACTIVATION VALUES)
			# k is before j
			derivativeActivation[0][:, i] = np.matmul(
				self.weights[1].transpose(),
				self.dActFunc(self.zLinear[1][:, i]) * derivativeActivation[1][:, i]
			)

			# --- Backpropagate layer 1-2 (calculate derivative with respect to WEIGHTS)
			gradientVector.weights[2][:, :] += np.outer(
				derivativeSoftmax(self.zLinear[2][:, i]) * derivativeActivation[2][:, i],
				self.neuronLayers[1][:, i]
			)

			# --- Backpropagate layer 0-1 (calculate derivative with respect to WEIGHTS)
			gradientVector.weights[1][:, :] += np.outer(
				self.dActFunc(self.zLinear[1][:, i]) * derivativeActivation[1][:, i],
				self.neuronLayers[0][:, i]
			)

			# --- Backpropagate layer 2 (calculate derivative with respect to BIASES)
			gradientVector.biases[2][:] += derivativeSoftmax(self.zLinear[2][:, i]) * derivativeActivation[2][:, i]

			# --- Backpropagate layer 1 (calculate derivative with respect to BIASES)
			gradientVector.biases[1][:] += self.dActFunc(self.zLinear[1][:, i]) * derivativeActivation[1][:, i]

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
		return sum(self.accuracies) / len(self.accuracies)


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
		biasDataFrame = pd.DataFrame(data=self.bias[1:, :, 0].transpose(),
			index=list(range(10)),
			columns=["layer1", "layer2"])
		biasDataFrame.to_csv(f"{folderName}/bias/bias.csv", encoding="utf-8")
		print("Written biases successfully!")


	# ----- Pandas read weights and biases from CSV file -----
	def readFromCSV(self, folderName):
		weights1 = np.array(pd.read_csv(f"{folderName}/weights/layer1.csv"))
		weights2 = np.array(pd.read_csv(f"{folderName}/weights/layer2.csv"))
		bias = np.array(pd.read_csv(f"{folderName}/bias/bias.csv"))

		self.weights[1] = weights1[:, 1:]
		self.weights[2] = weights2[:, 1:]
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

