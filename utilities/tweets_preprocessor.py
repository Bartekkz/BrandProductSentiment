import os
import pandas as pd
import numpy as np
import string
import nltk
import pickle
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from embeddings.EmbExtractor import EmbExtractor
from ekphrasis.classes.spellcorrect import SpellCorrector
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class tweetsPreprocessor(BaseEstimator, TransformerMixin):
    '''
    helper class to clean tweets, tokenzier tweets, create padded sequences
    based on ekphrasis which is a text processing tool 
    '''
    def __init__(self):
        self.preprocessor = self.create_preprocessor()


    def create_preprocessor(self):
        preprocessor = TextPreProcessor(
                normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'url', 'date', 'number'],
                annotate={"hashtag", "allcaps", "elongated",'emphasis', 'censored'},
                fix_html=True,
                segmenter='twitter',
                corrector='twitter',
                unpack_hashtags=True,
                unpack_contractions=True,
                spell_correct_elong=True,
                tokenizer=SocialTokenizer(lowercase=True).tokenize,
                dicts=[emoticons]
            )

        return preprocessor 


    def clean_tweets(self, txt):
        '''
        remove punctuation from tweet, encodes it
        '''
        txt = ''.join(v for v in txt if v not in string.punctuation)
        txt = txt.encode('utf8').decode('ascii', 'ignore')
        return txt


    def preprocess_tweets(self, tweets): 
        '''
        preprocesses tweets using TextPreProcessor from ekphrasis tool
        @params:
        :tweets: str -> tweet or list of tweets
        '''
        cleaned_tweets = []
        if isinstance(tweets, list):
            for tweet in tweets:
                clean_tweet = self.preprocessor.pre_process_doc(tweet)
                clean_tweet = ' '.join(word for word in clean_tweet)
                #clean_tweet = [word for word in clean_tweet.split() if word not in string.punctuation]
                cleaned_tweets.append(clean_tweet)
            return cleaned_tweets 
        else:
            clean_tweet = self.preprocessor.pre_process_doc(tweets)
            clean_tweet = ' '.join(word for word in clean_tweet)
            clean_tweet = [word for word in clean_tweet.split() if word not in string.punctuation]
            return [clean_tweet]


    def transform(self, X, y=None):
        path = 'data/tweets/pickled/processed_tweets.pickle' 

        if os.path.exists(path):
            with open(path, 'rb') as f:
                processed_tweets = pickle.load(f)

        else:
            processed_tweets = self.preprocess_tweets(X)
            with open(path, 'wb') as f:
                pickle.dump(processed_tweets, f)

        return processed_tweets 


    def fit(self, X, y=None):
        return self
