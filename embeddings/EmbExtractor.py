
import numpy as np

class EmbExtractor:
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
        for word in text:
            if word in self.word_idxs:
                print('GOTIT')
                idx_text.append(self.word_idxs[word])
            else:
                print(word)
                print('UNKOWN')
                if self.unk_policy == 'random':
                    idx_text.append(self.word_idxs['<unk>'])
                elif self.unk_policy == 'zero':
                    idx_text.append(0)
        return idx_text
    
    @staticmethod
    def pad_seq(tokenized_text, maxlen, padding='post'):
        if isinstance(tokenized_text, list):
            tokenized_text = np.asarray(tokenized_text)
        if tokenized_text.ndim == 1:
            tokenized_text = np.asarray([tokenized_text])
        padded_seq = np.zeros((len(tokenized_text), maxlen), dtype='int32')
        for i, text in enumerate(tokenized_text):
            if text.shape[0] < maxlen:
                if padding == 'pre':
                    padded_seq[i] = np.pad(text, (0, maxlen - len(text)), 'constant')
                elif padding == 'post':
                    padded_seq[i] = np.pad(text, (maxlen - len(text), 0), 'constant')
            elif text.shape[0] > maxlen:
                padded_seq = text[:maxlen]
        return padded_seq


    def get_padded_seq(self, text, padding='pre'):
        tokenized_text = self.tokenize_text(text)
        if self.maxlen > 0:
            padded_seq = self.pad_seq(tokenized_text, self.maxlen, padding=padding) 
            return padded_seq
        return tokenized_text