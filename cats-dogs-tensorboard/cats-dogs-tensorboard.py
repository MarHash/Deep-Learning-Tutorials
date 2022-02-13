import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import numpy as np
import time

# we add a name with time to easily identify the name in tensorboard
MODEL_NAME = f"cats-dogs-cnn-64x2-{int(time.time())}"

# we create a tensorboard log. NOTE: GPU is required for tensorboard
tensorboard = TensorBoard(log_dir=f'logs/{MODEL_NAME}')

# we load our data generated from the previous notebook
X = pickle.load(open(r"C:\Users\Onsor\OneDrive\Desktop\Deep learning tutorial\X.pickle", "rb")) # alread a numpy array
y = pickle.load(open(r"C:\Users\Onsor\OneDrive\Desktop\Deep learning tutorial\y.pickle", "rb"))
y = np.array(y)

# we will normalize the data manually as we know a pixel value is 0 to 255
X = X/255.0

model = Sequential()

# first we add a convolution of input size 64, with filer 3x3, and the size
# of the whole dataset, then follow this with activation and pooling layers
model.add(Conv2D(64, (3,3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2))) # pooling for downsampling, usually 2x2

# next is the second convolution layers
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# we need to flatten the data before adding a dense layer to 1d since our data is 2d
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

# our outpur is binart, so one neuron only for the output layer
model.add(Dense(1))
model.add(Activation("sigmoid")) # sigmoid is good for binary classification

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

model.fit(X, y, epochs=15, batch_size=32, validation_split=0.1, callbacks=[tensorboard]) # we let 10% of the data for validation