#!/usr/bin/env python3

import os
import numpy as np
import string
import warnings
from keras.utils import to_categorical
from keras.layers import LSTM
from utilities.tweets_preprocessor import tweetsPreprocessor
from utilities.data_loader import load_data, load_training_data, get_embeddings, load_train_test 
from models.rnn_model import build_attention_rnn
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

MAXLEN = 50
CORPUS = 'datastories.twitter'
DIM = 300

if __name__ == '__main__':    
    X_train, X_test, y_train, y_test, tokenizer = load_train_test(MAXLEN) 
    emb_matrix = get_embeddings(CORPUS, DIM, tokenizer) 
    model = build_attention_rnn(
        emb_matrix,
        classes=3,
        maxlen=MAXLEN,
        unit=LSTM,
        layers=2,
        trainable_emb=False,
        bidirectional=True,
        attention='simple',
        dropout_attention=0.5,
        layer_dropout_rnn=0.3,
        dropout_rnn=0.3,
        rec_dropout_rnn=0.3,
        clipnorm=1,
        lr=0.001,
        loss_l2=0.0001
    )    

    print(model.summary())
    print('Training model...')
    model.fit(X_train,
              y_train,
              validation_data=(X_test, y_test),
              epochs=50,
              batch_size=50
              ) 
    print('Model trained!')
    print('Saving weights...')
    model.save_weights('./data/model_weights/bi_model_weights_1.h5')
    print('Done!')






'''
TODO:
  - change helper.clean_tweets function with re

'''
