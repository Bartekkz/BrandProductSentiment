
import time
import os
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from embeddings.word_vectors_manager import WordVectorsManager
from utilities.tweets_preprocessor import tweetsPreprocessor

def load_data():
  df = pd.read_csv('~/Downloads/judge-1377884607_tweet_product_company.csv',
                                        usecols=['tweet_text', 'sentiment'],
                                        encoding='latin-1')

  df['sentiment'] = df.sentiment.apply(lambda x: 'neutral' if x == 'No emotion toward brand or product'
                                                           else 'positive' if x == 'Positive emotion'
                                                           else 'negative' if x == 'Negative emotion'
                                                           else np.nan)
  df = df.dropna()
  df['sentiment'] = df.sentiment.map({'negative':-1, 'neutral':0, 'positive':1})
  return df


def load_training_data(num_samples, divide=True):
  files_path = '../data/downloaded/'
  data = []
  for fname in os.listdir(os.path.abspath(files_path)):
    if 'new_' in fname:
      df = pd.read_csv(os.path.join(files_path, fname), encoding='utf-8')
      df = df[['sentiment', 'tweet_text']]
      df['sentiment'] = df.sentiment.map({'negative':-1, 'neutral':0, 'positive':1})
      df.dropna()
      data.append(df)
    else:
      continue
  data = pd.concat(data, ignore_index=True)
  if num_samples > len(data):
    data = data.sample(frac=1)
  else:
    data = data.sample(num_samples)
  if divide:
    tweets = data.tweet_text.tolist()
    labels = data.sentiment.values
    return tweets, labels
  return data


def get_embeddings(tweets, corpus, dim):
  preprocessor = tweetsPreprocessor()
  input_seq, tokenizer = preprocessor.tokenize_tweets(tweets)
  curr_time = time.time()
  emb_dict = WordVectorsManager('../data/', corpus=corpus, dim=dim, omit_non_english=True).read()
  emb_matrix = np.zeros((10000, dim))
  missed = 0
  for word, i in tokenizer.word_index.items():
    if i < 10000:
      vect = emb_dict.get(word) 
      if vect is not None:
        emb_matrix[i] = vect
      else:
        missed += 1
    else:
      break
  delta = time.time() - curr_time
  if delta < 60:
    print(f'Loading embeddings took: {delta} seconds')
  else:
    print(f'Loading embeddings took: {int(delta/60)} minutes')
  print(f'Missed words: {missed}')
  return emb_matrix
  

def name_cols_in_training_data():
        files_path = '../data/downloaded/'
        for fname in os.listdir(files_path):
                df = pd.read_csv(os.path.join(files_path, fname), delimiter='\t', encoding='utf-8', header=None)
                df.rename(columns={0:'number', 1:'sentiment', 2:'tweet_text'}, inplace=True)
                df.to_csv(os.path.join(files_path, str('new_' + fname)), index=False)
        print('Dataframe column names are changed!')



