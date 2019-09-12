#!/usr/bin/env python3

import os
import numpy as np
import string
import warnings
from keras.utils import to_categorical
from keras.layers import LSTM, Bidirectional, Dense, Embedding 
from kutilities.layers import Attention
from keras.models import Sequential, load_model 
from utilities.tweets_preprocessor import tweetsPreprocessor
from utilities.data_loader import load_data, load_training_data, get_embeddings, load_train_test 
from models.rnn_model import build_attention_rnn
from sklearn.model_selection import train_test_split

# ignore warnings from libriaries
warnings.filterwarnings('ignore')

#Constants
MAXLEN = 50
CORPUS = 'datastories.twitter'
DIM = 300

preprocessor = tweetsPreprocessor(MAXLEN)

if __name__ == '__main__':    

  X_train, X_test, y_train, y_test, tokenizer = load_train_test(MAXLEN) 
  emb_matrix = get_embeddings(CORPUS, DIM, tokenizer) 
  model = build_attention_rnn(
      emb_matrix,
      classes=1,
      maxlen=MAXLEN,
      unit=LSTM,
      layers=2,
      trainable_emb=False,
      bidirectional=True,
      attention='simple',
      dropout_attention=0.5,
      layer_dropout_rnn=0.3,
      dropout_rnn=0.5,
      rec_dropout_rnn=0.5,
      clipnorm=1,
      lr=0.01,
      loss_l2=0.0001
  )    
  print(model.summary())
  print('Training model...')
  model.fit(X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=18,
            batch_size=128
            ) 
  print('Model trained!')
  print('Saving model...')
  model.save(os.path.join(os.path.abspath('data/model_weights'), 'bi_model_4.h5'))
  print('Done!')

  model = load_model('./data/model_weights/bi_model_4.h5', custom_objects={'Attention':Attention()})
  pad, _ = preprocessor.get_padded_seq('I hate you bitch not bad sad')
  
  prediction = model.predict(pad)
  print(prediction)
