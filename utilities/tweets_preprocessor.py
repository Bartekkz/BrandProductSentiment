import os
import pandas as pd
import numpy as np
import string
import nltk
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.classes.spellcorrect import SpellCorrector
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


class tweetsPreprocessor:
    '''
    helper class to clean tweets, tokenzier tweets, create padded sequences
    based on ekphrasis which is a text processing tool 
    '''
    def __init__(self, maxlen):
        '''
        @params:
        :maxlen: int -> max length of the input sequences
        '''
        self.preprocessor = self.create_preprocessing_pipeline()
        self.maxlen = maxlen


    def create_preprocessing_pipeline(self, normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'], annotate={"hashtag", "allcaps", "elongated",
        'emphasis', 'censored'}, fix_html=True, segmenter="twitter", corrector="twitter", unpack_hashtags=True,
        unpack_contractions=True, spell_correct_elong=True, tokenizer=SocialTokenizer(lowercase=True).tokenize,
        dicts=[emoticons]):
        '''
        create text processing pipeline 
        @returns:
        :ekphrasis.classes.preprocessor.TextPreProcessor object
        '''
        text_processor = TextPreProcessor(
            normalize=normalize,
            annotate=annotate,
            fix_html=fix_html,
            spell_correct_elong=spell_correct_elong,
            unpack_contractions=True,
            unpack_hashtags=unpack_hashtags,
            tokenizer=tokenizer,
            spell_correction=True,
            corrector='english',
            segmenter=segmenter,
            dicts=dicts
        )
        return text_processor


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
                clean_tweet = [word for word in clean_tweet.split() if word not in string.punctuation]
                cleaned_tweets.append(clean_tweet)
            return cleaned_tweets 
        else:
            clean_tweet = self.preprocessor.pre_process_doc(tweets)
            clean_tweet = ' '.join(word for word in clean_tweet)
            clean_tweet = [word for word in clean_tweet.split() if word not in string.punctuation]
            return [clean_tweet]


    def tokenize_tweets(self, tweets, labels=None, num_classes=3):
        '''
        return tokenized array of words using keras Tokenizer   
        @params:
        :tweets: str/array -> twitter tweets 
        :labels: array -> array of labels for given tweets
        :num_classes: int -> number of output classes eg. (negative, neutral, positivee) - > 3
        @return:
        :input_seq -> array of tokenized words
        :tokenizer: keras.preprocessing.text.Tokenizer object fitted on given tweets
        '''
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(tweets)
        input_seq = tokenizer.texts_to_sequences(tweets)
        if labels is not None: 
            #labels = to_categorical(labels, num_classes)
            return input_seq, labels, tokenizer
        return input_seq, tokenizer 


    def get_padded_seq(self, tweets, labels=None, padding='pre', preprocess=True):
        '''
        pad to sequences given tweets
        @params:
        :tweeets: str/array - > twitter tweets 
        :labels: array -> array of labels for given tweets
        :padding: str -> padding of zeros in padded_seq
        :preprocess: bool -> whenever You want to preprocess tweets with self.preprocess_tweets 
        functions
        @return:
        padded sequences, binary class matrix(labels), tokenizer
        '''
        if preprocess:
            tweets = self.preprocess_tweets(tweets)
            if labels is not None:
                input_seq, labels, tokenizer = self.tokenize_tweets(tweets, labels)
                pad = pad_sequences(input_seq, maxlen=self.maxlen, padding=padding)
                return pad, labels, tokenizer
            else:
                input_seq, tokenizer = self.tokenize_tweets(tweets)
        else:
            input_seq = tweets 
        pad = pad_sequences(input_seq, maxlen=self.maxlen, padding=padding)
        return pad, tokenizer



        
