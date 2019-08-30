#!/usr/bin/env python3

from keras.utils import to_categorical
from network import Network
import os
import numpy as np
import string
import warnings
from utilities.tweets_preprocessor import tweetsPreprocessor
from utilities.data_loader import load_data, load_training_data, get_embeddings
warnings.filterwarnings('ignore')

preprocessor = tweetsPreprocessor()



if __name__ == '__main__':
  tweets, labels = load_training_data()
  tweets = tweets
  tweets = preprocessor.preprocess_tweets(tweets)
  tweets = preprocessor.tokenize_tweets(tweets)
  print(tweets[0:2])
  #vectors = get_embeddings('glove.twitter.27B', 200)
  #print(vectors.keys())








'''
TODO:
  - change helper.clean_tweets function with re

'''
