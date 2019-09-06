
import time
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
    files_path = './data/tweets/'
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


def load_train_test(maxlen: int):
    preprocessor = tweetsPreprocessor(maxlen)
    tweets, labels = load_training_data(10000)
    pad, labels, tokenizer = preprocessor.get_padded_seq(tweets, labels)   
    X_train, X_test, y_train, y_test = train_test_split(pad,
                                                        labels,
                                                        test_size=0.3,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test, tokenizer


def get_embeddings(corpus, dim, tokenizer):
    word_index = tokenizer.word_index
    vectors = WordVectorsManager(os.path.join(os.path.abspath('.'), 'embeddings'), corpus, 300).read()
    vocab_size = len(vectors)
    print(f'Loaded {vocab_size} vectors')
    # Create embeddings matrix
    emb_matrix = np.ndarray(shape=(len(vectors) + 1, dim), dtype='float32')
    for word, i in word_index.items():                
        emb_vector = vectors.get(word)
        if emb_vector is not None:
            emb_matrix[i] = emb_vector

    return emb_matrix 
    

def name_cols_in_training_data():
        files_path = './data/tweets'
        for fname in os.listdir(files_path):
            df = pd.read_csv(os.path.join(files_path, fname), delimiter='\t', encoding='utf-8', header=None)
            df.rename(columns={0:'number', 1:'sentiment', 2:'tweet_text'}, inplace=True)
            df.to_csv(os.path.join(files_path, str('new_' + fname)), index=False)
        print('Dataframe column names are changed!')

