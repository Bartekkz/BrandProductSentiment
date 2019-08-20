from sklearn import preprocessing
from keras.layers import Embedding, Bidirectional, LSTM, RNN 
from keras.models import Model, Sequential


class Network:
	def embedding_layer(self, embeddings, maxlen, trainable=False, masking=False,
	scale=False, normalize=False):
		if scale:
			embeddings = preprocessing.scale(embeddings)
		if normalize:
			embeddings = preprocessing.normalize(embeddings)

		vocab_size = embeddings.shape[0]
		embedding_size = embeddings.shape[1]

		_embedding = Embedding(
			input_dim=vocab_size,
			output_dim=embedding_size,
			input_length=maxlen if maxlen > 0 else None,
			trainable=trainable,
			mask_zero=masking if maxlen > 0 else False,
			weights=[embeddings]
		)

		return _embedding

	def get_rnn_layer(self, layer_type=LSTM, cells=64, bi=False, return_sequences=True,
	dropout=0., recurrent_dropout=0., l2_reg=0):
		rnn = layer_type(cells, return_sequences=return_sequences, dropout=dropout, 
		recurrent_dropout=recurrent_dropout)

		if bi:
			return (Bidirectional(rnn))
		else:
			return rnn	

	def build_attention_rnn(self, embeddings, classes, maxlen, layer_type=LSTM,
	cells=64, layers=1, **kwargs):
		bi = kwargs.get('bidirectional', False)
		dropout_rnn = kwargs.get('dropout_rnn', 0)
		rec_dropout_rnn = kwargs.get('rec_dropout_rnn', 0)
		dropout_attention = kwargs.get('dropout_attention', 0)
		attention = kwargs.get('attention', None)
		output_layer = kwargs.get('output_layer', False)
		clipnorm = kwargs.get('clipnorm', 0)
		loss_l2 = kwargs.get('loss_l2', 0.)
		lr = kwargs.get('lr', 0.001)

		input_layer = self.embedding_layer(embeddings=embeddings, max_length=maxlen, trainable=False,
		masking=True, scale=False, normalize=False)