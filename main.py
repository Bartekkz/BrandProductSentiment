import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv('~/Downloads/judge-1377884607_tweet_product_company.csv', usecols=['tweet_text', 'sentiment'], encoding='latin-1')

    df['sentiment'] = df.sentiment.apply(lambda x: 'neutral' if x == 'No emotion toward brand or product' 
                                             else 'positive' if x == 'Positive emotion' 
                                             else 'negative' if x == 'Negative emotion' 
                                             else np.nan)
    df = df.dropna()
    df['sentiment'] = df.sentiment.map({'negative':-1, 'neutral':0, 'positive':1})
    return df


if __name__ == '__main__':
    data = load_data()
    print(data.sample(2))