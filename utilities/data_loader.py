import time
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from embeddings.word_vectors_manager import WordVectorsManager
from utilities.tweets_preprocessor import tweetsPreprocessor


def load_data():
    '''
    loads test data from Downloads folder
    so You need to place your file there
    '''
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


def load_training_data(num_samples=0, divide=True):
    '''
    loads trainig data from data/tweets folder
    @params:
    :num_samples: int -> num of samples to random choose from dataframe(all by default)
    :divide: bool -> whenever You want to divide dataframe into list of tweets and labels
    or simply return dataFrame object
    '''
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
    if num_samples > len(data) or num_samples == 0:
        data = data.sample(frac=1)
    else:
        data = data.sample(num_samples)
    if divide:
        tweets = data.tweet_text.tolist()
        labels = data.sentiment.values
        return tweets, labels
    return data


def load_train_test(maxlen: int, num_samples=0):
    '''
    loads, preprocesses and splits training data into train and test sets
    @params:
    :maxlen: int -> max lenght of the input sequence
    :num_samples: int -> number of samples to use from dataframe(all by default)
    '''
    preprocessor = tweetsPreprocessor(maxlen)
    tweets, labels = load_training_data(num_samples)
    pad, labels, tokenizer = preprocessor.get_padded_seq(tweets, labels)   
    X_train, X_test, y_train, y_test = train_test_split(pad,
                                                        labels,
                                                        test_size=0.3,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test, tokenizer


def get_embeddings(corpus, dim):
    '''
    load pretrained word_embeddings learn on twitter datastories
    @params:
    :corpus: str -> name of the file without dimension ("ex. twitter.datastories.300d.txtx" -> 
    "twitter.datastories")
    :dim: int -> dimension of the embeddings matrix
    '''
    vectors = WordVectorsManager(os.path.join(os.path.abspath('.'), 'embeddings'), corpus, 300).read()
    print(f'Loaded {len(vectors)} vectors')
    position = 0
    word_map = {}
    # Create embeddings matrix
    emb_matrix = np.ndarray(shape=(len(vectors) + 2, dim), dtype='float32')
    for i, (word, vector) in enumerate(vectors.items()):
        if len(vector) > dim-100:
            position = i + 1
            word_map[word] = position 
            emb_matrix[position] = vector
    # add unknown token
    position += 1
    word_map['<unk>'] = position
    emb_matrix[position] = np.random.uniform(-0.05, 0.05, size=dim)
    return emb_matrix, word_map  

def name_cols_in_training_data():
    '''
    by default trainig data has no column names
    add column names to the training dataframe
    ''' 
    files_path = './data/tweets'
    for fname in os.listdir(files_path):
        df = pd.read_csv(os.path.join(files_path, fname), delimiter='\t', encoding='utf-8', header=None)
        df.rename(columns={0:'number', 1:'sentiment', 2:'tweet_text'}, inplace=True)
        df.to_csv(os.path.join(files_path, str('new_' + fname)), index=False)
    print('Dataframe column names are changed!')

        

        
