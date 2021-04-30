# app.py
from flask import Flask, redirect, url_for, request, render_template

# import sys
import os
# import glob
# import re
# import numpy as np
# import tensorflow as tf
# import pandas as pd

import io

# from keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from keras.models import load_model
# from keras.preprocessing import image
# from tensorflow import keras

import tflite_runtime.interpreter as tflite

# Flask utils
# from flask import Flask, redirect, url_for, request, render_template
# from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

# from PIL import Image, ImageChops
from PIL import Image
# import cv2


app = Flask(__name__)


# @app.route('/')
# def index():
#     return "<h1>HOLA STYLEIDc !!</h1>"


@app.route('/')
def index():
    # Main page
    # s = io.BytesIO(
    # b'https://styleidam.s3-us-west-1.amazonaws.com/model.tflite_model1')
    # model_path='https://styleidam.s3-us-west-1.amazonaws.com/model.tflite_model1'
    # interpreter = tf.lite.Interpreter(
    #     model_path='https://styleidam.s3-us-west-1.amazonaws.com/model.tflite_model1')
    # return render_template('index.html')
    # data = pd.read_csv(
    #     'https://styleidam.s3-us-west-1.amazonaws.com/prueba.csv')
    # model = tf.keras.models.load_model(
    # 'https://styleidam.s3-us-west-1.amazonaws.com/model_baseline01')
    # interpreter = tf.lite.Interpreter(model_content=s)
    # interpreter.allocate_tensors()
    return "<h1>m</h1>"


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
