import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from csvloader import *
from neural import *


NUM_BATCHES = 10000 // BATCH_SIZE
FOLDER_NAME = "3b1b_nn"

def constrainedInput(prompt, answers, inputFunc=(lambda s : s)):
	ans = inputFunc(input(prompt))
	while ans not in answers:
		ans = inputFunc(input(prompt))
	return ans

FOLDER_NAME = constrainedInput("NN folder name (3b1b_nn/temp_nn): ", ["3b1b_nn", "temp_nn"])
NN_TYPE = constrainedInput("NN type (sigmoid/relu): ", ["sigmoid", "relu"])


def main():
	MNIST_CSV_LOADER_OBJECT = MNIST_CSV_Loader()

	if NN_TYPE == "sigmoid":
		NEURAL_NETWORK = NeuralNetwork(sigmoid, derivativeSigmoid, "XAVIER", 0.1)
	elif NN_TYPE == "relu":
		NEURAL_NETWORK = NeuralNetwork(ReLU, derivativeReLU, "HE", 0.01)
	# NEURAL_NETWORK = NeuralNetwork(leakyReLU, derivativeLeakyReLU, "HE", 0.01)

	# ----- Read from old neural network
	NEURAL_NETWORK.readFromCSV(FOLDER_NAME)



	for _ in range(NUM_BATCHES):
		testingExampleBatch = MNIST_CSV_LOADER_OBJECT.getTestBatch()
		if testingExampleBatch is None:
			break

		NEURAL_NETWORK.forwardPropagation(testingExampleBatch)



	print(f"Weights[1]: {NEURAL_NETWORK.weights[1]}")
	print(f"Weights[2]: {NEURAL_NETWORK.weights[2]}")
	print(f"Bias[1]: {NEURAL_NETWORK.bias[1]}")
	print(f"Bias[2]: {NEURAL_NETWORK.bias[2]}")
	print(f"Overall accuracy: {NEURAL_NETWORK.overallAccuracy()}")





	# ----- Plotting
	x = np.array(list(range(len(NEURAL_NETWORK.accuracies))))
	y = np.array(NEURAL_NETWORK.accuracies)
	plt.scatter(x, y)
	plt.show()





if __name__ == "__main__":
	main()
