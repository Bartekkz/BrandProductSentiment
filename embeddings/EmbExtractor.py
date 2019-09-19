
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
np.seterr(all='ignore')
import os
from sklearn.base import BaseEstimator, TransformerMixin
from utilities import tweets_preprocessor
from ekphrasis.classes import preprocessor
from ekphrasis.classes.spellcorrect import SpellCorrector
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons




class EmbExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, word_idxs, maxlen=0, unk_policy='random', **kwargs):
        self.word_idxs = word_idxs
        self.maxlen = maxlen
        self.unk_policy = unk_policy


    def tokenize_text(self, texts):
        tokenized_words = [] 
        for text in texts:  
            tokenized_words.append(np.asarray(self.idx_text(text)))
        return np.asarray(tokenized_words)


    def idx_text(self, text):
        idx_text = []
        for word in text.split():
            if word in self.word_idxs:
                idx_text.append(self.word_idxs[word])
            else:
                if self.unk_policy == 'random':
                    idx_text.append(self.word_idxs['<unk>'])
                elif self.unk_policy == 'zero':
                    idx_text.append(0)
        return idx_text


    def pad_seq(self, tokenized_text, maxlen, padding='pre'):
        if isinstance(tokenized_text, list): 
            tokenized_text = np.asarray(tokenized_text)
        padded_seq = np.zeros((len(tokenized_text), maxlen), dtype='int32') 
        for i, text in enumerate(tokenized_text):
            if text.shape[0] < maxlen:
                if padding == 'pre':
                    padded_seq[i] = np.pad(text, (0, maxlen - len(text)), 'constant')
                elif padding == 'post':
                    padded_seq[i] = np.pad(text, (maxlen - len(text), 0), 'constant')
            elif text.shape[0] > maxlen:
                padded_seq[i] = text[:maxlen]
        return padded_seq


    def get_padded_seq(self, text, padding='pre'):
        if isinstance(text, str):
            print('Swaping to list...')
            text = [text]
        tokenized_text = self.tokenize_text(text) 
        if self.maxlen > 0:
            padded_seq = self.pad_seq(tokenized_text, self.maxlen, padding=padding) 
            return padded_seq
        return tokenized_text


    def transform(self, X, y=None):
        print('Transforming...')
        padded_seq = self.get_padded_seq(X)
        return padded_seq


    def fit(self, X, y=None):
        return self



