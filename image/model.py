"""
Module contains code of convolutional neural network.
Only showcase purpose.

Referring to this source:
    - https://www.geeksforgeeks.org/applying-convolutional-neural-network-on-mnist-dataset/
"""


import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow import keras

EPOCHS = 5
n_classes = 10
img_rows = 28
img_cols = 28

# load training and testing data
(x_train, Y_train), (x_test, Y_test) = mnist.load_data()

# preprocess
if keras.backend.image_data_format() == 'channels_first':
   x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
   x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
   inpx = (1, img_rows, img_cols)
 
else:
   x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
   x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
   inpx = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

Y_train = tf.keras.utils.to_categorical(Y_train, n_classes)
Y_test = tf.keras.utils.to_categorical(Y_test, n_classes)


# define model
model = tf.keras.models.Sequential([
  
        # Input convolution layer with 32 Filters using 3x3 Kernel and batch normalization
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(img_rows, img_cols, 1)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Dropout(0.4),
        
        # Flatten
        tf.keras.layers.Flatten(),

        # Hidden layer with Dropout
        tf.keras.layers.Dense(256, activation="relu"),

        # Output Layer
        tf.keras.layers.Dense(10, activation="softmax")
    ])


# compile model
model.compile(
    optimizer='adam',
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# train model
model.fit(
    x_train, Y_train,
    epochs=EPOCHS,
    batch_size=32
)

#model.save("number_recognition.h5")