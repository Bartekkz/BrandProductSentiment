from sklearn import preprocessing
from keras.layers import Embedding, Bidirectional, LSTM, RNN, Dropout, Dense, Activation
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.constraints import max_norm
from kutilities.layers import Attention, AttentionWithContext, MeanOverTime


def embedding_layer(embeddings, maxlen, trainable=False, masking=False,
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
    mask_zero=masking if maxlen > 0 else False
    )

    return _embedding

def get_rnn_layer(layer_type=LSTM, cells=64, bi=False, return_sequences=True,
        dropout=0., recurrent_dropout=0., implementation=1,  l2_reg=0):
    rnn = layer_type(cells, return_sequences=return_sequences, dropout=dropout, 
    recurrent_dropout=recurrent_dropout, implementation=implementation, 
    kernel_regularizer=l2(l2_reg))

    if bi:
        return (Bidirectional(rnn))
    else:
        return rnn  

def build_attention_rnn(embeddings, classes, maxlen, layer_type=LSTM,
                                                cells=64, layers=1, **kwargs): 
    trainable_emb = kwargs.get('trainable_emb', False)
    bi = kwargs.get('bidirectional', False)
    layer_dropout_rnn = kwargs.get('layer_dropout_rnn', 0)
    dropout_rnn = kwargs.get('dropout_rnn', 0)
    rec_dropout_rnn = kwargs.get('rec_dropout_rnn', 0)
    dropout_attention = kwargs.get('dropout_attention', 0)
    attention = kwargs.get('attention', None)
    dropout_final = kwargs.get('dropout_final', 0)
    fc1 = kwargs.get('fc1', False)
    clipnorm = kwargs.get('clipnorm', 0)
    loss_l2 = kwargs.get('loss_l2', 0.)
    lr = kwargs.get('lr', 0.001)
    
    print('Creating model...')
    # init the model
    model = Sequential()
    model.add(embedding_layer(embeddings=embeddings, maxlen=maxlen, trainable=trainable_emb, masking=True)) 

    for i in range(layers):
        return_seq = (layers > 1 and i < layers - 1) or attention
        model.add(get_rnn_layer(layer_type, cells, bi, return_sequences=return_seq, dropout=dropout_rnn,
            recurrent_dropout=rec_dropout_rnn))
        if layer_dropout_rnn > 0:
            model.add(Dropout(layer_dropout_rnn))

    if attention == 'memmory':
        model.add(AttentionWithContext())
        if dropout_attention > 0:
            model.add(Dropout(dropout_attention))
    elif attention == 'simple':
        model.add(Attention())
        if dropout_attention > 0:
            model.add(Dropout(dropout_attention))
    
    if fc1:
        model.add(Dense(100))
        if dropout_final > 0:
            model.add(Dropout(dropout_final))

    model.add(Dense(classes, activity_regularizer=l2(loss_l2)))
    model.add(Activation('softmax'))

    model.compile(optimizer=Adam(clipnorm=clipnorm, lr=lr),
                                loss='categorical_crossentropy')
    return model
