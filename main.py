#!/usr/bin/env python3

from helper import Helper
import string
import warnings
warnings.filterwarnings('ignore')
helper = Helper()


if __name__ == '__main__':
	data = helper.load_data()    
	sample_data = data.tweet_text.sample(2)
	cleaned_data, tweets = helper.clean_tweets(sample_data)

	for line in tweets:
		print(line)
	for line in cleaned_data:
		print(line)