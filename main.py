#!/usr/bin/env python3

from helper import Helper
from keras.utils import to_categorical
import string
import warnings
warnings.filterwarnings('ignore')
helper = Helper()


if __name__ == '__main__':
	data = helper.load_data()    
	sample_data = data.sample(5)
	sample_tweets = sample_data.tweet_text
	labels = sample_data.sentiment
	clean_text = helper.preprocess_tweets(sample_tweets)
	input_seq, total_words = helper.tokenize_tweets(clean_text)
	padded_seq = helper.get_padded_seq(input_seq)

