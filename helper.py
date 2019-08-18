import pandas as pd
import numpy as np
import string
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from keras.preprocessing.text import Tokenizer


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

	def create_preprocessing_pipeline(self, normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
		'time', 'url', 'date', 'number'], annotate={"hashtag", "allcaps", "elongated", "repeated",
		'emphasis', 'censored'}, fix_html=True, segmenter="twitter", corrector="twitter", unpack_hashtags=True,
		unpack_contractions=True, spell_correct_elong=True, tokenizer=SocialTokenizer(lowercase=True),
		dicts=[emoticons]):
		text_processor = TextPreProcessor(
			unpack_contractions=unpack_contractions,
			normalize=normalize,
			annotate=annotate,
			fix_html=fix_html,
			segmenter=segmenter,
			corrector=corrector,
			unpack_hashtags=unpack_hashtags,
			spell_correct_elong=spell_correct_elong,
			spell_correction=True,
			tokenizer=tokenizer.tokenize,
			dicts=dicts,
			remove_tags=True
		)
		return text_processor


	def tokenize_txt(self, txt):
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(txt)
		corpus = tokenizer.texts_to_sequences(txt)
		return corpus

	def clean_text(self, txt):
		string_punc = [char for char in string.punctuation if char not in ['#', '{', '}', "'"]]
		txt = ''.join(v for v in txt if v not in string_punc).lower()
		txt = txt.encode('utf8').decode('ascii', 'ignore')
		return txt

	def clean_tweets(self, tweets):
		text_processor = self.create_preprocessing_pipeline()
		tweets =	[self.clean_text(tweet) for tweet in tweets]
		cleaned_text = []
		
		for tweet in tweets:
			clean_tweet = text_processor.pre_process_doc(tweet)
			cleaned_text.append(clean_tweet)
			
		return cleaned_text, tweets

	def pad_sequences(self):
		pass


    