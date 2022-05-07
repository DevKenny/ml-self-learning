import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras import optimizers
from keras import losses

# region Load dataset

# MNIST = Modified National Institute of Standards and Technology
# Is a large database of handwritten digits
from tensorflow.python.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('X Train', x_train.shape)
print('Y Train', y_train.shape)
# endregion

# region Process datasets
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
# endregion

# region Create model
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.adam_v2.Adam(),
              metrics=['accuracy'])
# endregion

# region Train the model'
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=6, epochs=3)
print('The model ha successfully trained')

model.save('mnist.h5')
print('Saving the model mnist.h5')
# endregion

# region Evaluate the model

loss, accuracy = model.evaluate(x_test, y_test)
print('Test loss', loss)
print('Test accuracy', accuracy)

# endregion
