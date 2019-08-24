#!/usr/bin/env python3
import numpy as np
import pytest
import warnings
from helper import Helper
warnings.filterwarnings('ignore')


class TestHelper:
    def test_helper_clean_tweets_for_a_single_tweet(self):
        input_tweet = 'this game sucks! should i sell it? im out...'
        helper = Helper()
        cleaned_tweet = helper.clean_tweets(input_tweet)
        expected = 'this game sucks should i sell it im out'
        assert cleaned_tweet == expected
    
    def test_helper_clean_tweets_for_a_several_tweets(self):
        input_tweets = ['hello?', 'what is! up?'] 
        helper = Helper()
        cleaned_tweet = [helper.clean_tweets(tweet) for tweet in input_tweets] 
        expected = ['hello', 'what is up'] 
        assert cleaned_tweet == expected

    def test_helper_preprocess_tweets_for_a_single_tweet(self):
        # fix for single tweet
        input_tweet = 'HeLLo, that game sucks! #game :)'
        helper = Helper()
        preprocessed_tweet = helper.preprocess_tweets(input_tweet)
        expected = ['hello that game sucks <hashtag> game </hashtag> <happy>'] 
        assert preprocessed_tweet == expected
    
    def test_helper_preprocess_tweets_for_several_tweets(self):
        input_tweets = ['HeLLO this Game! sucks !', 'What Is UP??']
        helper = Helper()
        preprocessed_tweets = helper.preprocess_tweets(input_tweets)
        expected = [['hello', 'this', 'game', 'sucks'], ['what', 'is', 'up']]
        assert preprocessed_tweets == expected

    def test_helper_tokenize_tweets(self):
        input_string = 'hello my Friend! #life'
        helper = Helper()
        cleaned_tweet = helper.preprocess_tweets(input_string)
        input_seq = helper.tokenize_tweets(cleaned_tweet)
        expected = [[2, 3, 4, 1, 5, 1]]
        assert input_seq == expected

    def test_helper_get_padded_seq(self):
        input_string = ['hello?', 'What is wrong with you!']
        helper = Helper()
        cleaned_tweets = helper.preprocess_tweets(input_string)
        input_seq = helper.tokenize_tweets(cleaned_tweets)
        padded = helper.get_padded_seq(input_seq)
        expected = np.array([[0, 0, 0, 0, 1], [2, 3, 4, 5, 6]])
        assert padded.all() == expected.all()
        

        