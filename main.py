#!/usr/bin/env python3

from helper import Helper
from keras.utils import to_categorical
from network import Network
import numpy as np
import string
import warnings
warnings.filterwarnings('ignore')
helper = Helper()


if __name__ == '__main__':
    #data = helper.load_data()    
    #sample_data = data.sample(5)
    #sample_tweets = sample_data.tweet_text
    #labels = sample_data.sentiment
    #clean_text = helper.preprocess_tweets(sample_tweets)
    #input_seq, total_words = helper.tokenize_tweets(clean_text)
    #padded_seq = helper.get_padded_seq(input_seq)
    #glove_model = helper.loadGloveModel()
    #print(glove_model['hello'])
    tweet = ['hello! WhAt is goin? on #angel :)', 'fuck u hehe :(']
    tweet = helper.preprocess_tweets(tweet)
    input_seq = helper.tokenize_tweets(tweet)
    tweet1 = 'hello what is Goin> on #angel :)'
    tweet1 = helper.preprocess_tweets(tweet1)
    seq = helper.tokenize_tweets(tweet1)
    helper.get_padded_seq(input_seq, 12, preprocess=False)
    helper.get_padded_seq(seq, 12, preprocess=False)
    helper.get_padded_seq('hello what is Goin> on #angel :)', 12, preprocess=True)
    helper.get_padded_seq(['hello! WhAt is goin? on #angel :)', 'fuck u hehe :('], 12, preprocess=True)






'''
TODO:
  - change helper.clean_tweets function with re  

'''
