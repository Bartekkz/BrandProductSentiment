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

#preprocessor = tweetsPreprocessor()
MAX_LENGHT = 50
CORPUS = 'datastories.twitter'
DIM = 300

if __name__ == '__main__':
    embeddings, word_indices = get_embeddings(CORPUS, DIM) 
    print(word_indices)    







'''
TODO:
  - change helper.clean_tweets function with re

'''
