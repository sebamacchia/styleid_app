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
from tensorflow import keras

# Flask utilss
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

MODEL_PATH = 'models/MobileNet_full_wikiart_R2.h5'

# Load your trained model
model = load_model(MODEL_PATH, compile=False)


print('Model loaded. Check http://127.0.0.1:5000/')

class_names = ['Abstract Art', 'Baroque', 'Cubism',
               'High Renaissance', 'Impressionism', 'Pop Art']

# class_names = ['abstract-art',
#               #  'abstract-expressionism',
#               #  'academicism',
#               #  'art-informel',
#               #  'art-nouveau',
#                'baroque',
#               #  'color-field-painting',
#                'cubism',
#               #  'early-renaissance',
#                'expressionism',
#                'high-renaissance',
#                'impressionism',
#               #  'late-renaissance',
#               #  'magic-realism',
#               #  'naive-art-primitivism',
#               #  'neoclassicism',
#               #  'northern-renaissance',
#                'pop-art',
#               #  'post-impressionism',
#               'realism',
#               'rococo',
#               'romanticism',
#               'surrealism',
#               #  'symbolism',
#               'ukiyo-e',
#  ]

# from PIL import Image, ImageChops
# # F_IN = "/content/gdrive/MyDrive/QUEESTILOES/dataset/train/28363.jpg"
# # F_OUT = "/content/impresionista11.jpg"
# F_IN = "/content/Cuadro1.jpg"
# F_OUT = "/content/Cuadro11.jpg"
# size = (256,256)
# image = Image.open(F_IN)
# image.thumbnail(size, Image.ANTIALIAS)


# image_size = image.size
# thumb = image.crop((0, 0, size[0], size[1]))
# offset_x = int(max((size[0] - image_size[0]) / 2, 0))
# offset_y = int(max((size[1] - image_size[1]) / 2, 0))
# thumb = ImageChops.offset(thumb, offset_x, offset_y)
# thumb.save(F_OUT)


def model_predict(img_path, model):

    img = image.load_img(img_path, target_size=(224, 224))
    print('esta haciendo la prediccion')

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = (x - np.min(x)) / (np.max(x) - np.min(x))

    preds = model.predict(x)

    print(f'la prediccion es: {preds}')

    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        print('cargo la imagen')
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

        print(f'el score fue de {score}')

        print(str(format(class_names[np.argmax(score)])))

        return str(format(class_names[np.argmax(score)]))
        # return print(
        #     "This image most likely belongs to {} with a {:.2f} percent confidence."
        #     .format(class_names[np.argmax(score)], 100 * np.max(score))
        # )
    return None


if __name__ == '__main__':
    app.run(debug=True)
