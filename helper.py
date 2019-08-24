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


class Helper:
	def load_data(self):
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

	def loadGloveModel(self, dimension='100'):
		print('Loading glove model...')
		model = {}
		with open(f'../data/datastories.twitter.{dimension}d.txt', 'r') as f:
			for line in f:
				splitLine = line.split()
				word = splitLine[0]
				embedding = np.array([float(val) for val in splitLine[1:]])
				model[word] = embedding
		print(f'Done! {len(model)} words loaded!')
		return model

	def load_training_data(self):
		files_path = '../data/downloaded/'
		data = []
		for fname in os.listdir(os.path.join(files_path)):
			if 'new_' in fname:
					df = pd.read_csv(os.path.join(files_path, fname), encoding='utf-8')
					df = df[['sentiment', 'tweet_text']]
					df['sentiment'] = df.sentiment.map({'negative':-1, 'neutral':0, 'positive':1})
					df.dropna()
					data.append(df)
			else:
				continue
		data = pd.concat(data, ignore_index=True)
		return data


	def name_cols_in_training_data(self):
		files_path = '../data/downloaded/'
		for fname in os.listdir(files_path):
			df = pd.read_csv(os.path.join(files_path, fname), delimiter='\t', encoding='utf-8', header=None)
			df.rename(columns={0:'number', 1:'sentiment', 2:'tweet_text'}, inplace=True)
			df.to_csv(os.path.join(files_path, str('new_' + fname)), index=False)
		print('Dataframe column names are changed!')

	def create_preprocessing_pipeline(self, normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
		'time', 'url', 'date', 'number'], annotate={"hashtag", "allcaps", "elongated", "repeated",
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

	def tokenize_tweets(self, txt):
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(txt)
		total_words = len(tokenizer.word_index) + 1
		input_seq = tokenizer.texts_to_sequences(txt)
		return input_seq, total_words

	def clean_tweets(self, txt):
		string_punc = [char for char in string.punctuation if char not in ['#', '{', '}', "'", ':', ')', '(']]
		txt = ''.join(v for v in txt if v not in string_punc)
		txt = txt.encode('utf8').decode('ascii', 'ignore')
		return txt

	def preprocess_tweets(self, tweets):
		text_processor = self.create_preprocessing_pipeline()
		cleaned_tweets = []
		if type(tweets) == list:
			tweets = [self.clean_tweets(tweet) for tweet in tweets]
			for tweet in tweets:                
				clean_tweet = text_processor.pre_process_doc(tweet)
				cleaned_tweets.append(clean_tweet)
			return cleaned_tweets
		else:
				tweet = self.clean_tweets(tweets)
				clean_tweet = text_processor.pre_process_doc(tweet)
				return clean_tweet	
    
	def get_padded_seq(self, input_seq):
		maxlen = max([len(seq) for seq in input_seq])
		padded_seq =	pad_sequences(input_seq, maxlen=maxlen, padding='pre') 
		return padded_seq




    