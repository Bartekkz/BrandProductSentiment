
import numpy as np
import time
from sklearn import preprocessing
from keras.layers import Embedding, Bidirectional, LSTM, RNN, Dropout, Dense, Activation
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.constraints import max_norm
from kutilities.layers import Attention, AttentionWithContext, MeanOverTime
from sklearn.pipeline import Pipeline
from utilities.data_loader import get_embeddings
from utilities.tweets_preprocessor import tweetsPreprocessor
from embeddings.EmbExtractor import EmbExtractor
from keras.models import load_model


def embedding_layer(embeddings, maxlen, trainable=False, masking=False,
    scale=False, normalize=False):
    '''
    create Keras based embedding layer
    @params:
    :embeddings: array - >embeddings weights matrix
    :maxlen: int -> max lenght of the input sequence 
    :trainable: bool -> whenever You want to update weight in given layer
    :scale: bool -> scale weights 
    :normalize: bool -> normalize weights
    @return:
    :keras.layers.Embedding object
    '''
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
    '''
    creates rnn layer of a given type
    if bi(bool) is True returns
    bidirectional layer of given type
    '''

    if bi:
        return (Bidirectional(rnn))
    else:
        return rnn  

def build_attention_rnn(embeddings, classes, maxlen, layer_type=LSTM,
                                                cells=64, layers=1, **kwargs): 
    '''
    creates rnn based model
    @params:
    :embeddings: array-> embeddigns matrix
    :classes: int -> num of label classes 
    :maxlen: int -> max lenght of the input sequence
    :layer_type: keras.layers -> type of rnn layer
    :cells: int -> amount of cells in a single layer
    :**kwargs: params like all kind of dropouts etc.
    @returns:
    :keras.model.sequential object
    '''
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
    model.add(embedding_layer(embeddings=embeddings, maxlen=maxlen, trainable=trainable_emb, masking=True, scale=True)) 

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


def predict(tweet, model_weights='data/model_weights/new_bi_model_2.h5'):
    curr_time = time.time()
    MAXLEN = 50 
    CORPUS = 'datastories.twitter'
    DIM = 300
    _, word_map = get_embeddings(CORPUS, DIM)  
    pipeline = Pipeline([
        ('preprocessor', tweetsPreprocessor(load=False)),
        ('extractor', EmbExtractor(word_idxs=word_map, maxlen=MAXLEN))
    ])
    pad = pipeline.transform(tweet) 
    model = load_model(model_weights, custom_objects={'Attention':Attention()})
    prediction = model.predict(pad)
    for pred in prediction:
        if np.argmax(pred) == 2:
            print('negative')
        elif np.argmax(pred) == 1:
            print('positive')
        else:
            print('neutral')
    delta = time.time() - curr_time
    print(f'Predicting took: {delta} seconds')
