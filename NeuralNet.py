import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import pickle
import time


moload = True

NAME = "Cats-vs-dog-cnn-64x2-{}".format(int(time.time()))

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

y = np.array(y)

X = X / 255.0

print(type(y))
print(type(X))

if moload == False:
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])

    model.fit(X, y, batch_size=32, epochs=5000, validation_split=0.3)

    model.save('64x3-CNN.model')

else:
    model = tf.keras.models.load_model("64x3-CNN.model")
    model.fit(X, y, batch_size=32, epochs=300, validation_split=0.3)
    model.save('64x3-CNN.model')