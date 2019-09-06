#!/usr/bin/env python3

import os
import numpy as np
import string
import warnings
from keras.utils import to_categorical
from keras.layers import LSTM
from utilities.tweets_preprocessor import tweetsPreprocessor
from utilities.data_loader import load_data, load_training_data, get_embeddings, load_train_val_test 
from models.rnn_model import build_attention_rnn
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

MAXLEN = 50
CORPUS = 'datastories.twitter'
DIM = 300

if __name__ == '__main__':    
    X_train, X_test, y_train, y_test, tokenizer = load_train_val_test(MAXLEN) 
    emb_matrix = get_embeddings(CORPUS, DIM, tokenizer) 
    print(type(emb_matrix))
    print(len(emb_matrix))

    '''
    load_train_val_test(MAXLEN) 
    nn_model = build_attention_rnn(
    embeddings,
    classes=3,
    maxlen=MAXLEN,
    unit=LSTM
    )    
    '''






'''
TODO:
  - change helper.clean_tweets function with re

'''
