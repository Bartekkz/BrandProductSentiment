#!/usr/bin/env python3

import os
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf

from flask import Flask, render_template, request, redirect, url_for
import pandas as pd   
import requests
import json
from keras.models import load_model
from utilities.tweets_preprocessor import tweetsPreprocessor
from utilities.data_loader import get_embeddings
from embeddings.EmbExtractor import EmbExtractor
from models.rnn_model import predict
from kutilities.layers import Attention
import numpy as np

np.random.seed(44)


app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('landingPage.html')


@app.route('/api/', methods=['POST'])
def predict_tweet():
  data = request.get_json()
  #with graph.as_default():
  #    prediction = predict(data, model) 
  return data 


@app.route('/end')
def end():
  return render_template('end.html')


@app.route('/analyze')
def analyze():
    return render_template('analyze.html', error='')


@app.route('/test')
def test():
    return render_template('test.html')


@app.route('/test', methods=['POST'])
def get_text():
  headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
  text = request.form['textInp']
  print(text)
  data = json.dumps(text)
  r = requests.post(url, data=data, headers=headers)
  return r.text 


@app.route('/read', methods=['POST', 'GET'])    
def read_csv():
    print('Reading...')
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    f = request.files.get('data_file')
    try:
        data = pd.read_csv(f)
        point = data.value[1]
        data = json.dumps(point)
        r = requests.post(url, data=data, headers=headers)
        print(r)
        print(r.text)
        return r.text 
    except:
        return render_template('analyze.html', error='You can only load csv files')
   

if __name__ == '__main__':
    url = 'http://localhost:5002/api/'
    #model_weights = 'data/model_weights/new_bi_model_2.h5'
    #model = load_model(model_weights, custom_objects={'Attention':Attention()})
    #global graph
    #graph = tf.get_default_graph()
    app.run(debug=True, host='localhost', port=5002)

'''
TODO:
    - finish route for predicting from csv file
    - create test analyzer route
    - add navbar
    - change text of error to sth like 'do not forget to upload csv file :)'
    - send model_weights to this laptop
'''
