import pandas as pd
import numpy as np
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons


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
		unpack_contractions=True, spell_correct_elong=False, tokenizer=SocialTokenizer(lowercase=True),
		dicts=[emoticons]):
		text_processor = TextPreProcessor(
			normalize=normalize,
			annotate=annotate,
			fix_html=fix_html,
			segmenter=segmenter,
			corrector=corrector,
			unpack_hashtags=unpack_hashtags,
			unpack_contractions=unpack_contractions,
			spell_correct_elong=spell_correct_elong,
			tokenizer=tokenizer.tokenize,
			dicts=dicts
		)
		return text_processor




	def clean_tweets(self, tweets):
		pass
    