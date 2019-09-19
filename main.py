#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
np.seterr(all='ignore')
import string
from keras.utils import to_categorical
from keras.layers import LSTM, Bidirectional, Dense, Embedding 
from kutilities.layers import Attention
from keras.models import Sequential, load_model 
from utilities.tweets_preprocessor import tweetsPreprocessor
from utilities.data_loader import load_data, load_training_data, get_embeddings, load_train_test 
from embeddings.EmbExtractor import EmbExtractor
from models.rnn_model import build_attention_rnn
from models.rnn_model import predict
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

np.random.seed(44)

#Constants
MAXLEN = 40 
CORPUS = 'datastories.twitter'
DIM = 300

if __name__ == '__main__':      
    predict(tweet=['Fuck you man I hate you!', 'I am so happy :)'])
    



    #model = load_model('./data/model_weights/new_bi_model_1.h5', custom_objects={'Attention':Attention()})
    #print(model.summary())
    '''
    tweet = ['Fuck you man I hate you bad sad hate shit :/', 'I love you I am so happy this is so good great :)',
    'this is the worst game i have ever play #shit', 'thanks man this is brilliant, wonderful game :)']
    tweet = preprocessor.preprocess_tweets(tweet)
    pad = extractor.get_padded_seq(tweet)
    for padded in pad:
        padded = np.array([padded])
        predictoin = model.predict(padded)
        if np.argmax(predictoin) == 2:
            print('negative')
        elif np.argmax(predictoin) == 1:
            print('positive')
        else:
            print('neutral')
    '''
'''
TODO:
    - fix labels order
    - create function to predict
    - deployment
'''


    

