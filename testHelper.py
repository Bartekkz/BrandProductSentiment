#!/usr/bin/env python3
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
        expected = ['hello', 'that', 'game', 'sucks', '<hashtag>', 'game', '</hashtag>', '<happy>'] 
        assert preprocessed_tweet == expected
    
    def test_helper_preprocess_tweets_for_several_tweets(self):
        input_tweets = ['HeLLO this Game! sucks !', 'What Is UP??']
        helper = Helper()
        preprocessed_tweets = helper.preprocess_tweets(input_tweets)
        expected = [['hello', 'this', 'game', 'sucks'], ['what', 'is', 'up']]
        assert preprocessed_tweets == expected