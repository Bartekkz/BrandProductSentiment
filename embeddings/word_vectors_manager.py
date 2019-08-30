import numpy as np
import errno
import os
import pickle

class WordVectorsManager:
  def __init__(self, data_path, corpus=None, dim=None, omit_non_english=True):
    self.data_path = data_path
    self.wv_fname = f'{corpus}.{str(dim)}d.txt'
    self.parsed_fname = f'{corpus}.{str(dim)}.pickle'
    self.corpus = corpus
    self.dim = dim
    self.omit_non_english = omit_non_english

  def is_ascii(self, text):
    try:
      text.encode('ascii')
      return True
    except:
      return False

  def write(self):
    _word_vector_file = os.path.join(os.path.abspath(self.data_path), self.wv_fname)
    if os.path.exists(_word_vector_file):
      embeddings_dict = {}
      with open(_word_vector_file, 'r') as f:
        for line in f:
          split_line = line.split()
          word = split_line[0]
          embedding = np.array([float(val) for val in split_line[1:]])

          if self.omit_non_english and not self.is_ascii(word):
                  continue
      
          embeddings_dict[word] = embedding
      f.close()       
      print(f'Found {len(embeddings_dict)} word vectors')

      with open(os.path.join(self.data_path, self.parsed_fname), 'wb') as pickle_file:
        pickle.dump(embeddings_dict, pickle_file)

    else:
      print(f'{_word_vector_file} not founded!')
      raise FileNotFoundError(
      errno.ENOENT, os.strerror(errno.ENOENT), _word_vector_file
    )

  def read(self):
    _parsed_file = os.path.join(self.data_path, self.parsed_fname)
    if os.path.exists(_parsed_file):
      with open(_parsed_file, 'rb') as f:
        return pickle.load(f)
    else:
      self.write()
      return self.read()

