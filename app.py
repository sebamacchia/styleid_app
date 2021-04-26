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
MODEL_PATH = 'models/model_777.h5'

# Load your trained model
model = load_model(MODEL_PATH, compile=False)

print('Model loaded. Check http://127.0.0.1:5000/')

class_names = ['Abstract Art', 'Baroque', 'Cubism',
               'High Renaissance', 'Impressionism', 'Pop Art']


def model_predict(img_path, model):

    img = image.load_img(img_path, target_size=(256, 256))
    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = (x - np.min(x)) / (np.max(x) - np.min(x))

    preds = model.predict(x)

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

    return None


if __name__ == '__main__':
    app.run(debug=True)
