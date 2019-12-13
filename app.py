from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import pickle

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

app = Flask(__name__)
model_pkl = pickle.load(open('model.pkl','rb'))

model_path = 'resnet152V2.h5'

model = load_model(model_path)
model._make_predict_function()

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):

    img = load_img(img_path,target_size=(128,128))
    img = img_to_array(img)
    img = img.reshape(1,128,128,3)
    pred = model.predict(img)
    
    return pred


@app.route('/', methods=['GET'])
def index():
    return render_template('base.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)

        result = (np.argmax(preds))
       
        return classes[result]
        
    return 'None Matched'


if __name__ == '__main__':
    app.run(debug=True)
