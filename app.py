#!/usr/bin/env python3

import os
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf

from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd   
import requests
import json
from keras.models import load_model
from utilities.tweets_preprocessor import tweetsPreprocessor
from utilities.data_loader import get_embeddings
from embeddings.EmbExtractor import EmbExtractor
from kutilities.layers import Attention
from sklearn.pipeline import Pipeline
from models.rnn_model import predict
import numpy as np

np.random.seed(44)


app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('landingPage.html')


@app.route('/api/', methods=['POST', 'GET'])
def predict_tweet():
    data = request.get_json()
    if isinstance(data, str):
        return data
    '''
    CODE:
    with graph.as_default():
        prediction = predict(data, pipeline, model)  
    data = {'preds':prediction}
    return jsonify({'prediction':prediction})
    '''
    #TODO:
    #change predict method to sum up neutral, positive and negative tweet and return some number
    tweets = list(data.values())
    num = 0 
    for tweet in tweets:
        print(tweet)
        num += 1
    return jsonify(num) 


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
    data = json.dumps(text)
    r = requests.post(url, data=data, headers=headers)
    return r.text
    #render_template('end.html', data=r.text)


@app.route('/read', methods=['POST', 'GET'])    
def read_csv():
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    approved_col_names = ['tweets', 'text', 'tweet', 'value', 'values', 'tweet_text', 'text_tweet']
    f = request.files.get('data_file')
    try:
        data = pd.read_csv(f)
        for col_name in data.columns:
            if col_name in approved_col_names:
                final_col = col_name
                break
        point = data[final_col][0:32]
        data = point.to_json() 
        r = requests.post(url, data=data, headers=headers)
        return render_template('end.html', pos=70, neg=8, neu=22)
    except:
        print('fail')
        return render_template('analyze.html', error=f'Remember You can only load .csv file and it has to \
                contain one of the followings columns: {approved_col_names}')


if __name__ == '__main__':
    url = 'http://localhost:5002/api/'
    '''
    model_weights = os.path.abspath('data/model_weights/new_bi_model_1.h5')
    model = load_model(model_weights, custom_objects={'Attention':Attention()})
    global graph
    graph = tf.get_default_graph()
    MAXLEN = 50 
    CORPUS = 'datastories.twitter'
    DIM = 300
    _, word_map = get_embeddings(CORPUS, DIM)  
    pipeline = Pipeline([
        ('preprocessor', tweetsPreprocessor(load=False)),
        ('extractor', EmbExtractor(word_idxs=word_map, maxlen=MAXLEN))
    ])
   '''
    app.run(debug=True, host='localhost', port=5002)
   


#TODO:
'''
    -add css for all kinds of devices
    - add navbar    
'''









