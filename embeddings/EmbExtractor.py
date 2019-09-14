
import numpy as np

class EmbExtractor:
    def __init__(self, word_idxs, maxlens, unk_policy='random', **kwargs):
        self.word_idxs = word_idxs
        self.maxlens = maxlens
        self.unk_policy = unk_policy
    

    def tokenize_words(self, text):
        tokenized_words = []
        for word in 

    
    def idx_text(self, text):
        idx_text = []
        for word in text:
            if word in self.word_idxs:
                idx_text.append(self.word_idxs[word])

