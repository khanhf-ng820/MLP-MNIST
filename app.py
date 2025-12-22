import pygame
import numpy as np
import pandas as pd
import sys
from csvloader import *
from neural import *



CELL_SIZE = 10
canvasRect = pygame.Rect(0, 100, 28 * CELL_SIZE, 28 * CELL_SIZE)
buttonRect = pygame.Rect(40, 20, 200, 60)
width, height = 28 * CELL_SIZE, 28 * CELL_SIZE + 100

image = np.zeros((28, 28))
FOLDER_NAME = "3b1b_nn"


def pointInRect(point, rect):
	xp, yp = point
	x, y, w, h = rect
	return xp >= x and xp <= x + w and yp >= y and yp <= y + h


def main():
	pygame.init()
	screen = pygame.display.set_mode((width, height))


	running = True
	clock = pygame.time.Clock()


	pygame.font.init()
	my_font = pygame.font.SysFont(None, 60)
	text_surface = my_font.render("Clear", False, (0, 0, 0))


	NEURAL_NETWORK = NeuralNetwork(sigmoid, derivativeSigmoid, "XAVIER", 0.1)
	# NEURAL_NETWORK = NeuralNetwork(ReLU, derivativeReLU, "HE", 0.01)
	# NEURAL_NETWORK = NeuralNetwork(leakyReLU, derivativeLeakyReLU, "HE", 0.01)

	NEURAL_NETWORK.readFromCSV(FOLDER_NAME)



	while running:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False

		printResults = False

		mouse_x, mouse_y = pygame.mouse.get_pos()
		mouse_y -= 100
		cell_x, cell_y = mouse_x // CELL_SIZE, mouse_y // CELL_SIZE

		left_button, middle_button, right_button = pygame.mouse.get_pressed()

		if left_button:
			printResults = True
			for dx in [-1, 0, 1]:
				for dy in [-1, 0, 1]:
					x, y = cell_x + dx, cell_y + dy
					if x in range(28) and y in range(28):
						image[y, x] = max(1. if dx == 0 and dy == 0 else 1., image[y, x])

			if pointInRect(pygame.mouse.get_pos(), buttonRect):
				image.fill(0.)


		for i in range(28):
			for j in range(28):
				pygame.draw.rect(
					screen,
					(int(image[i, j] * 255.), int(image[i, j] * 255.), int(image[i, j] * 255.)),
					(j * CELL_SIZE, i * CELL_SIZE + 100, CELL_SIZE, CELL_SIZE),
					width=0
				)

		pygame.draw.rect(
			screen,
			(255, 255, 255),
			canvasRect,
			width=1
		)

		pygame.draw.rect(
			screen,
			(255, 255, 255),
			buttonRect,
			width=0
		)
		screen.blit(text_surface, buttonRect[:2])

		exampleBatch = MNIST_ExampleBatch(
			np.tile(image.flatten(), (BATCH_SIZE, 1)).transpose(),
			None
		)

		outputNeurons = NEURAL_NETWORK.forwardPropagationNormal(exampleBatch)[:, 0]

		if printResults:
			print(f"Output: {outputNeurons}")
			print(f"Prediction: {np.argmax(outputNeurons)}")



		pygame.display.flip()
		clock.tick(60)



	pygame.quit()
	sys.exit()





if __name__ == "__main__":
	main()
