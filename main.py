#!/usr/bin/env python3

import os
import numpy as np
import string
import warnings
from keras.utils import to_categorical
from keras.layers import LSTM
from utilities.tweets_preprocessor import tweetsPreprocessor
from utilities.data_loader import load_data, load_training_data, get_embeddings
from models.rnn_model import build_attention_rnn
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

preprocessor = tweetsPreprocessor()



if __name__ == '__main__':
  tweets, labels = load_training_data(10000)
  print(len(tweets))
  padded, labels = preprocessor.get_padded_seq(tweets=tweets, labels=labels, maxlen=50) 
  emb_matrix = np.zeros(shape=(10000, 300))
  print('creating model...' )
  model =  build_attention_rnn(embeddings=emb_matrix, classes=3, maxlen=50, layer_type=LSTM, cells=150,
      layers=2, bidirectional=True, layer_dropout_rnn=0.5, attention='simple', final_layer=False,
      dropout_final=0.5, dropout_attention=0.5, dropout_rnn=0.5, rec_dropout_rnn=0.5, clipnorm=1, lr=0.001, 
      loss_l2=0.0001)
  print(model.summary())
  x_train, x_test, y_train, y_test = train_test_split(padded, labels, test_size=0.3)
  print(y_train[0:5])
  '''
  history = model.fit(x_train, 
                      y_train,
                      validation_data=(x_train, y_train),
                      epochs=50,
                      batch_size=50) 
  model.save_weights('../data/my_model_weights_1.h5')
  '''
  print(x_train[0])
  print('DOne')







'''
TODO:
  - change helper.clean_tweets function with re

'''
