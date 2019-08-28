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
  get_embeddings('datastories.twitter', 100)










'''
TODO:
  - change helper.clean_tweets function with re  

'''
