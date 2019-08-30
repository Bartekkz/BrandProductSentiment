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


class tweetsPreprocessor:
  def __init__(self):
    self.preprocessor = self.create_preprocessing_pipeline()


  def create_preprocessing_pipeline(self, normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
    'time', 'url', 'date', 'number'], annotate={"hashtag", "allcaps", "elongated",
    'emphasis', 'censored'}, fix_html=True, segmenter="twitter", corrector="twitter", unpack_hashtags=True,
    unpack_contractions=True, spell_correct_elong=True, tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[emoticons]):
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
    txt = ''.join(v for v in txt if v not in string.punctuation)
    txt = txt.encode('utf8').decode('ascii', 'ignore')
    return txt


  def preprocess_tweets(self, tweets): 
    cleaned_tweets = []
    if type(tweets) == list:
      for tweet in tweets:                				
        clean_tweet = self.preprocessor.pre_process_doc(tweet)
        clean_tweet = ' '.join(word for word in clean_tweet)
        clean_tweet = [word for word in clean_tweet.split() if word not in string.punctuation]
        cleaned_tweets.append(clean_tweet)
      return cleaned_tweets 
    else:
      clean_tweet = text_processor.pre_process_doc(tweets)
      clean_tweet = ' '.join(word for word in clean_tweet)
      clean_tweet = [word for word in clean_tweet.split() if word not in string.punctuation]
      return [clean_tweet]

  def tokenize_tweets(self, tweets):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets)
    input_seq = tokenizer.texts_to_sequences(tweets)
    return input_seq 

def get_padded_seq(self, data, maxlen, padding='pre', preprocess=True):
  if preprocess:
    tweets = self.preprocess_tweets(data)
    input_seq = self.tokenize_tweets(tweets)
  else:
    input_seq = data
  pad = pad_sequences(input_seq, maxlen=maxlen, padding=padding)
  return pad



    
