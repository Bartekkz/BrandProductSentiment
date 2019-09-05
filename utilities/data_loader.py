
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


def get_embeddings(corpus, dim):
    vectors = WordVectorsManager(os.path.join(os.path.abspath('.'), 'embeddings'), corpus, 300).read()
    vocab_size = len(vectors)
    print(f'Loaded {vocab_size} vectors')
    wv_map = {}
    # Create embeddings matrix
    emb_matrix = np.ndarray(shape=(vocab_size + 2, dim), dtype='float32')
    for i, (word, vector) in enumerate(vectors.items()):
        if len(vector) > 199:
            pos = i + 1
            wv_map[word] = pos
            emb_matrix[pos] = vector
    pos += 1
    wv_map['<unk>'] = pos
    emb_matrix[pos] = np.random.uniform(-0.05, 0.05, dim)
    
    return emb_matrix, wv_map
def name_cols_in_training_data():
        files_path = '../data/downloaded/'
        for fname in os.listdir(files_path):
                df = pd.read_csv(os.path.join(files_path, fname), delimiter='\t', encoding='utf-8', header=None)
                df.rename(columns={0:'number', 1:'sentiment', 2:'tweet_text'}, inplace=True)
                df.to_csv(os.path.join(files_path, str('new_' + fname)), index=False)
        print('Dataframe column names are changed!')


