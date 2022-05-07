import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('mnist.h5')

image_number = 1
while os.path.isfile(f'digits/digit{image_number}.png'):
    try:
        img = cv2.imread('digits/digit{}.png'.format(image_number))[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print("The number is probably a {}".format(np.argmax(prediction)))

        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()

        image_number += 1
    except:
        print("Error reading image! Proceeding to the next one...")
        image_number += 1
