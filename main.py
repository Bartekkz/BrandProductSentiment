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
from kutilities.layers import Attention, AttentionWithContext
from keras.models import Sequential, load_model 
from utilities.tweets_preprocessor import tweetsPreprocessor
from utilities.data_loader import load_data, load_training_data, get_embeddings, load_train_test 
from embeddings.EmbExtractor import EmbExtractor
from models.rnn_model import build_attention_rnn
from models.rnn_model import predict
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

np.random.seed(44)

if __name__ == '__main__':       
    emb_matrix, word_map = get_embeddings('datastories.twitter', 300)

    pipeline = Pipeline([
        ('preprocessor', tweetsPreprocessor(load=True)),
        ('extractor', EmbExtractor(word_idxs=word_map, maxlen=50))])
    X_train, X_val, y_train, y_val = load_train_test(pipeline=pipeline, test_size=0.2)

    model = build_attention_rnn(
        emb_matrix,
        classes=3,
        maxlen=50,
        unit=LSTM,
        layers=2,
        trainable_emb=False,
        bidirectional=True,
        attention='simple',
        dropout_attention=0.5,
        layer_dropout_rnn=0.5,
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
              validation_data=(X_val, y_val),
              epochs=18,
              batch_size=128
              )
    print('Model trained')
    print('saving model...')
    model.save(os.path.join(os.path.abspath('data/model_weights'), 'new_bi_model_1.h5'))
    print('doone')
    del model
    model = load_model('data/model_weights/new_bi_model_1.h5')
    print(model.summary())


'''
TODO:
    - check source code of Attention layer to fix the problem
'''
    



