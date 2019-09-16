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
from embeddings.EmbExtractor import EmbExtractor
from models.rnn_model import build_attention_rnn
from sklearn.model_selection import train_test_split

# ignore warnings from libriaries
warnings.filterwarnings('ignore')

np.random.seed(44)

#Constants
MAXLEN = 25 
CORPUS = 'datastories.twitter'
DIM = 300


preprocessor = tweetsPreprocessor(MAXLEN)


if __name__ == '__main__':      
    X_train, X_test, y_train, y_test, _= load_train_test(MAXLEN) 
    emb_matrix, word_map = get_embeddings(CORPUS, DIM)
    extractor = EmbExtractor(word_map, MAXLEN)
    padded = extractor.get_padded_seq(['hello my name is John']) 
    print(padded)


    #model = build_attention_rnn(
    #    emb_matrix,
    #    classes=3,
    #    maxlen=MAXLEN,
    #    unit=LSTM,
    #    layers=2,
    #    trainable_emb=False,
    #    bidirectional=True,
    #    attention='simple',
    #    dropout_attention=0.5,
    #    layer_dropout_rnn=0.3,
    #    dropout_rnn=0.5,
    #    rec_dropout_rnn=0.5,
    #    clipnorm=1,
    #    lr=0.01,
    #    loss_l2=0.0001
    #)        
    #print(model.summary())
    #model.load_weights('./data/model_weights/model_weights_1.h5')
    #print('Done')

    #print('Training model...')
    #model.fit(X_train,
    #        y_train,
    #        validation_data=(X_test, y_test),
    #        epochs=18,
    #        batch_size=128
    #        ) 
    #print('Model trained!')
    #print('Saving model...')
    #model.save(os.path.join(os.path.abspath('data/model_weights'), 'bi_model_5.h5'))
    #print('Done!')
    
     
    #pad, _ = preprocessor.get_padded_seq('Love happy good great enjoy wonderful')
     
    #prediction = model.predict(pad)
    #print(prediction)
    
