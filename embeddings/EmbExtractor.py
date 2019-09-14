
import numpy as np

class EmbExtractor:
    def __init__(self, word_idxs, maxlens, unk_policy='random', **kwargs):
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
        for word in text:
            if word in self.word_idxs:
                idx_text.append(self.word_idxs[word])
            else:
                if self.unk_policy == 'random':
                    idx_text.append(self.word_idxs['<unk>'])
                elif self.unk_policy == 'zero':
                    idx_text.append(0)
        return idx_text
    
    @staticmethod
    def get_padded_seq(tokenized_text, maxlen, padding='post'):
        padded_seq = np.zeros((len(X), maxlen), dtype='int32')
        for i, text in enumerate(tokenized_text):
            if text.shape[0] < maxlen:
                padded_text = np.pad(text, (0, ))

