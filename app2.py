from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf


# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
# c
# MODEL_PATH = '/Users/juansebastianmacchia/Desktop/Deployment-Deep-Learning-Model-master/models/model_777.h5'
MODEL_PATH = '/models/tfLite_models/model.tflite_model1'

# Load your trained model
# model = load_model(MODEL_PATH, compile=False)

interpreter = tf.lite.Interpreter(model_path='/content/model.tflite')
interpreter.allocate_tensors()


print('Model loaded. Check http://127.0.0.1:5000/')

class_names = ['Abstract Art', 'Baroque', 'Cubism',
               'High Renaissance', 'Impressionism', 'Pop Art']

# from PIL import Image, ImageChops
# # F_IN = "/content/gdrive/MyDrive/QUEESTILOES/dataset/train/28363.jpg"
# # F_OUT = "/content/impresionista11.jpg"
# F_IN = "/content/Cuadro1.jpg"
# F_OUT = "/content/Cuadro11.jpg"
# size = (256,256)
# image = Image.open(F_IN)
# image.thumbnail(size, Image.ANTIALIAS)


# image_size = image.size
# thumb = image.crop( (0, 0, size[0], size[1]) )
# offset_x = int(max( (size[0] - image_size[0]) / 2, 0 ))
# offset_y = int(max( (size[1] - image_size[1]) / 2, 0 ) )
# thumb = ImageChops.offset(thumb, offset_x, offset_y)
# thumb.save(F_OUT)

def model_predict(img_path, model):
    # ____________________________________________________________________
    F_IN = img_path
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

    # Se normaliza la imagen
    img_array = (img_array - np.min(img_array)) / \
        (np.max(img_array) - np.min(img_array))

    # make the prediction
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    tflite_model_predictions = interpreter.get_tensor(
        output_details[0]['index'])
    # print("Prediction results shape:", tflite_model_predictions.shape)
    # prediction_classes = np.argmax(tflite_model_predictions, axis=1)
    # print(prediction_classes)
    # print(prediction_classes.shape)
    # print(tflite_model_predictions.shape)
    # predictions = model.predict(img_array)
    # print('prediccion: ',tflite_model_predictions)

# ____________________________________________________________________

    preds = tflite_model_predictions

    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        prediction = model_predict(file_path, model)
        score = tf.nn.softmax(prediction[0])

        return str(format(class_names[np.argmax(score)]))
        # return print(
        #     "This image most likely belongs to {} with a {:.2f} percent confidence."
        #     .format(class_names[np.argmax(score)], 100 * np.max(score))
        # )
    return None


if __name__ == '__main__':
    app.run(debug=True)
