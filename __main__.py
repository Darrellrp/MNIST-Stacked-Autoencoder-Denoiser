import random
from time import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import optimizers
from keras.layers import Dense, Input
from keras.models import Sequential, Model
from keras.callbacks import TensorBoard

# Input 
MNIST_IMG_DIM = [28, 28]

# Autocoder
HIDDEN_SIZE = 128
CODE_SIZE = 32

# Training
EPOCHS = 5
BATCH_SIZE = 32
VERBOSE = 1

# Optimization
LEARNING_RATE = 0.001
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = None
DECAY = 0.0
AMSGRAD = False

# Post Training
EVALUATE = False
PREDICT = True
NUMBER_SHOWN_OF_PREDICTIONS = 4

# Load mnist dataset and generate train & test dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train_shape = x_train.shape[0]
input_size = np.prod(MNIST_IMG_DIM)

# Reshape datasets from (_, 28, 28) to (_, 784)
x_train = x_train.reshape(x_train.shape[0], input_size)
x_test = x_test.reshape(x_test.shape[0], input_size)

# Normalize RGB-values
x_train = np.true_divide(x_train, 255)
x_test = np.true_divide(x_test, 255)

# Convert to float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Input layer (size=748)
input_img = Input(shape=(input_size, ))

# Autoencoder 1.
hidden_1 = Dense(HIDDEN_SIZE, activation='relu')(input_img)
code_1 = (Dense(CODE_SIZE, activation='relu'))(hidden_1)

# Autoencoder 2.
hidden_2 = Dense(HIDDEN_SIZE, activation='relu')(code_1)
code_2 = (Dense(CODE_SIZE, activation='relu'))(hidden_2)

# Autoencoder 3.
hidden_3 = Dense(HIDDEN_SIZE, activation='relu')(code_2)
code_3 = (Dense(CODE_SIZE, activation='relu'))(hidden_3)

# Autoencoder 2.
hidden_4 = Dense(HIDDEN_SIZE, activation='relu')(code_3)
code_4 = (Dense(CODE_SIZE, activation='relu'))(hidden_4)

# Output layer (Reconstructed image, siz=748)
output_img = Dense(input_size, activation='sigmoid')(code_4)

autoencoder = Model(input_img, output_img)

adam = optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON, decay=DECAY, amsgrad=AMSGRAD)
autoencoder.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

autoencoder.fit(x_train, x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE, callbacks=[tensorboard])

if EVALUATE:
	score, accuracy = autoencoder.evaluate(x_test, x_test)
	print('Test score:', score)
	print('Test accuracy:', accuracy)

if PREDICT:
	reconstructed = autoencoder.predict(x_test)

	# Reshape datasets from (_, 784) to (_, 28, 28)
	reconstructed = reconstructed.reshape(x_test.shape[0], MNIST_IMG_DIM[0], MNIST_IMG_DIM[1])
	actual = x_test.reshape(x_test.shape[0], MNIST_IMG_DIM[0], MNIST_IMG_DIM[1])

	# Show a number of predictions
	for i in range(NUMBER_SHOWN_OF_PREDICTIONS):
		# Generate random index
		rand_i = random.randrange(0, x_test.shape[0])

		print('')

		# Print actual label
		print('The label is %d' % y_test[rand_i])

		# Show actual image
		plt.imshow(actual[rand_i], cmap='Greys')
		plt.show()

		# Show reconstructed image
		plt.imshow(reconstructed[rand_i], cmap='Greys')
		plt.show()
