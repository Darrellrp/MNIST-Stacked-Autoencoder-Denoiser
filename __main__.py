import tensorflow as tf
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.models import Sequential
import numpy as np
import random

EVALUATE = False
PREDICT = True

# Load mnist dataset and generate train & test dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train_shape = x_train.shape[0]
input_shape = [28, 28]
input_size = np.prod(input_shape)

# Reshape datasets from (_, 28, 28) to (_, 784)
x_train = x_train.reshape(x_train.shape[0], input_size)
x_test = x_test.reshape(x_test.shape[0], input_size)

# Normalize RGB-values
x_train = np.true_divide(x_train, 255)
x_test = np.true_divide(x_test, 255)

# Convert to float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

hidden_size = 128
code_size = 32

autoencoder = Sequential()

# Input layer (size=748) & Hidden layer 1
autoencoder.add(Dense(hidden_size, activation='relu', input_dim=input_size))
# Code layer
autoencoder.add(Dense(code_size, activation='relu'))
# Hidden layer 2
autoencoder.add(Dense(hidden_size, activation='relu'))
# Output layer (Reconstructed image, siz=748)
autoencoder.add(Dense(input_size, activation='sigmoid'))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

autoencoder.fit(x_train, x_train, epochs=5, batch_size=32, verbose=1)

if EVALUATE:
    score, accuracy = autoencoder.evaluate(x_test, x_test)
    print('Test score:', score)
    print('Test accuracy:', accuracy)


if PREDICT:
    reconstructed = autoencoder.predict(x_test)

    # Reshape datasets from (_, 784) to (_, 28, 28)
    reconstructed = reconstructed.reshape(x_test.shape[0], input_shape[0], input_shape[1])
    actual = x_test.reshape(x_test.shape[0], input_shape[0], input_shape[1])

    NUMBER_SHOWN_OF_PREDICTIONS = 4

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

