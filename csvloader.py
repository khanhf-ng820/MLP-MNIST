import numpy as np
import pandas as pd
import csv


IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_SIZE_TUPLE = (IMAGE_WIDTH, IMAGE_HEIGHT)
BATCH_SIZE = 20



class MNIST_ExampleBatch:
	def __init__(self, image, label):
		self.images = image
		self.labels = label

	def desiredOutputs(self):
		# Returns 2D np array
		# (One-hot desired outputs over all examples in batch)
		Y = np.zeros((10, BATCH_SIZE))
		for i in range(BATCH_SIZE):
			correctIndex = int(self.labels[i])
			Y[correctIndex, i] = 1
		return Y



class MNIST_CSV_Loader:
	def __init__(self):
		self.testFilePath = "mnist_test.csv"
		self.trainFilePath = "mnist_train.csv"

		# Read CSV into DataFrames
		self.testingData = pd.read_csv(self.testFilePath)
		self.trainingData = pd.read_csv(self.trainFilePath)
		# print(self.trainingData)

		# Turn DataFrames into NumPy arrays
		self.testingData = np.array(self.testingData, dtype=np.float64)
		self.trainingData = np.array(self.trainingData, dtype=np.float64)
		# print(self.trainingData)

		# Normalize pixels
		self.testingData[:, 1:] /= 255.0
		self.trainingData[:, 1:] /= 255.0

		# Transpose so that each column is an example
		self.testingData = self.testingData.transpose()
		self.trainingData = self.trainingData.transpose()

		# Separate labels from image pixels
		self.testingLabels = self.testingData[0, :]
		self.trainingLabels = self.trainingData[0, :]

		# Convert to int8 type
		self.testingLabels = self.testingLabels.astype(np.int8)
		self.trainingLabels = self.trainingLabels.astype(np.int8)

		# Separate labels from image pixels
		self.testingData = self.testingData[1:, :]
		self.trainingData = self.trainingData[1:, :]

		# Starting index of next mini-batch
		self.trainingExampleIndex = 0
		self.testingExampleIndex = 0

	def getTrainBatch(self):
		# Returns a MNIST_ExampleBatch of np arrays for training
		startIndex = self.trainingExampleIndex
		if startIndex > 60000 - BATCH_SIZE:
			return None
		self.trainingExampleIndex += BATCH_SIZE

		return MNIST_ExampleBatch(
			self.trainingData[:, startIndex:(startIndex + BATCH_SIZE)],
			self.trainingLabels[startIndex:(startIndex + BATCH_SIZE)]
		)

	def getTestBatch(self):
		# Returns a MNIST_ExampleBatch of np arrays for testing
		startIndex = self.testingExampleIndex
		if startIndex > 10000 - BATCH_SIZE:
			return None
		self.testingExampleIndex += BATCH_SIZE

		return MNIST_ExampleBatch(
			self.testingData[:, startIndex:(startIndex + BATCH_SIZE)],
			self.testingLabels[startIndex:(startIndex + BATCH_SIZE)]
		)

	def resetStart(self):
		self.trainingExampleIndex = 0
		self.testingExampleIndex = 0




