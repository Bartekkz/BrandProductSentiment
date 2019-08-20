from sklearn import preprocessing
from keras.layers import Embedding, Bidirectional, LSTM, RNN, Dropout, Dense, Activation
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.constraints import max_norm
from kutilities.layers import Attention, AttentionWithContext, MeanOverTime

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
	dropout=0., recurrent_dropout=0., implementation=1,  l2_reg=0):
		rnn = layer_type(cells, return_sequences=return_sequences, dropout=dropout, 
		recurrent_dropout=recurrent_dropout, implementation=implementation, 
		kernel_regularizer=l2(l2_reg))

		if bi:
			return (Bidirectional(rnn))
		else:
			return rnn	

	def build_attention_rnn(self, embeddings, classes, maxlen, layer_type=LSTM,
	cells=64, layers=1, **kwargs):
		bi = kwargs.get('bidirectional', False)
		layer_dropout_rnn = kwargs.get('layer_dropout_rnn', 0)
		dropout_rnn = kwargs.get('dropout_rnn', 0)
		rec_dropout_rnn = kwargs.get('rec_dropout_rnn', 0)
		dropout_attention = kwargs.get('dropout_attention', 0)
		attention = kwargs.get('attention', None)
		dropout_final = kwargs.get('dropout_final', 0)
		fc1 = kwargs.get('fc1', False)
		output_layer = kwargs.get('output_layer', False)
		clipnorm = kwargs.get('clipnorm', 0)
		loss_l2 = kwargs.get('loss_l2', 0.)
		lr = kwargs.get('lr', 0.001)

		input_layer = self.embedding_layer(embeddings=embeddings, maxlen=maxlen, trainable=False,
		masking=True, scale=False, normalize=False)

		# add layers with 'return_sequences' = True without output layer
		for layer in range(layers):
			return_seq = (layer > 1 and layer < layers - 1) or attention
			if layer == 0:
				rnn_layer = self.get_rnn_layer(layer_type=layer_type, cells=cells,
				bi=bi, return_sequences=return_seq, dropout=dropout_rnn,
				recurrent_dropout=rec_dropout_rnn, l2_reg=loss_l2)(input_layer)
				if layer_dropout_rnn > 0:
					dropout_layer = Dropout(layer_dropout_rnn)(rnn_layer)
			else:
				if layer_dropout_rnn > 0:
					rnn_layer = self.get_rnn_layer(layer_type=layer_type, cells=cells,
					bi=bi, return_sequences=return_seq, dropout=dropout_rnn,
					recurrent_dropout=rec_dropout_rnn, l2_reg=loss_l2)(dropout_layer)
					dropout_layer = Dropout(layer_dropout_rnn)(rnn_layer)
				else:
					rnn_layer = self.get_rnn_layer(layer_type=layer_type, cells=cells,
					bi=bi, return_sequences=return_seq, dropout=dropout_rnn,
					recurrent_dropout=rec_dropout_rnn, l2_reg=loss_l2)(rnn_layer)

		if attention == 'memory':
			if layer_dropout_rnn > 0:
				attention_layer = AttentionWithContext()(dropout_layer)
				if dropout_attention > 0:
					dropout_attention_layer = Dropout(dropout_attention)(attention_layer)
			else:
				attention_layer = AttentionWithContext()(rnn_layer)
				if dropout_attention > 0:
					dropout_attention_layer = Dropout(dropout_attention)(attention_layer)
		elif attention == 'simple':
			if layer_dropout_rnn > 0:
				attention_layer = Attention()(dropout_layer)
				if dropout_attention > 0:
					dropout_attention_layer = Dropout(dropout_attention)(attention_layer)
			else:
				attention_layer = Attention()(rnn_layer)
				if dropout_attention > 0:
					dropout_attention_layer = Dropout(dropout_attention)(attention_layer)

		if fc1:
			if layer_dropout_rnn > 0:
				fc1 = Dense(100, kernel_constraint=max_norm(2.))(dropout_attention_layer)
				if dropout_final > 0:
					dropout_fc1_layer = Dropout(dropout_final)(fc1)
			else:
				fc1 = Dense(100, kernel_constraint=max_norm(2.))(attention_layer)
				if dropout_final > 0:
					dropout_fc1_layer = Dropout(dropout_final)(fc1)
		if dropout_final > 0:
			final_layer = Dense(classes, activity_regularizer=l2(loss_l2))(dropout_fc1_layer)
		else:
			final_layer = Dense(classes, activity_regularizer=l2(loss_l2))(fc1)

		output_layer = Activation(activation='softmax')(final_layer)

		model = Model(input_layer, output_layer)
		model.compile(optimizer=Adam,
					loss='categorical_crossentropy')
		return model