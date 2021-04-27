# importamos
import pandas as pd
import numpy as np
import os
import requests
import json

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import regularizers, optimizers
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow import keras
from PIL import Image, ImageChops

interpreter = tf.lite.Interpreter(
    model_path='/Users/juansebastianmacchia/Desktop/__GITHUB/styleid_app/models/tfLite_models/model.tflite_model1')

interpreter.allocate_tensors()


F_IN = "/Users/juansebastianmacchia/Desktop/__GITHUB/styleid_app/uploads/pop_art_01.jpg"
size = (256, 256)
image = Image.open(F_IN)
image.thumbnail(size, Image.ANTIALIAS)
image_size = image.size
thumb = image.crop((0, 0, size[0], size[1]))
offset_x = int(max((size[0] - image_size[0]) / 2, 0))
offset_y = int(max((size[1] - image_size[1]) / 2, 0))
img = ImageChops.offset(thumb, offset_x, offset_y)

# Convertimos la imagen en un array
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Se normaliza la imagen
img_array = (img_array - np.min(img_array)) / \
    (np.max(img_array) - np.min(img_array))

interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()
tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
print("Prediction results shape:", tflite_model_predictions.shape)
prediction_classes = np.argmax(tflite_model_predictions, axis=1)
print(prediction_classes)
print(prediction_classes.shape)
print(tflite_model_predictions.shape)

# Realizamos la predicción

class_names1 = ['Abstract Art', 'Baroque', 'Cubism',
                'High Renaissance', 'Impressionism', 'Pop Art']

# score1 = tf.nn.softmax(prediction_classes[0])

# Imprimimos la predicción
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names1[np.argmax(tflite_model_predictions)], 100 * np.max(tflite_model_predictions))
)
print(tflite_model_predictions)
