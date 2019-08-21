import spacy
import numpy as np
import pandas as pd
import cupy

from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, Embedding, Bidirectional
from keras.layers import TimeDistributed
from keras.optimizers import Adam
from keras.utils import to_categorical

def compile_lstm(embeddings, shape, settings):
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape["max_length"],
            trainable=False,
            weights=[embeddings],
            mask_zero=True,
        )
    )
    model.add(TimeDistributed(Dense(shape["nr_hidden"], use_bias=False)))
    model.add(
        Bidirectional(
            LSTM(
                shape["nr_hidden"],
                recurrent_dropout=settings["dropout"],
                dropout=settings["dropout"],
            )
        )
    )
    model.add(Dense(shape["nr_class"], activation="sigmoid"))
    model.compile(
        optimizer=Adam(lr=settings["lr"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def get_embeddings(vocab, using_gpu=True):
    if using_gpu:
        return cupy.asnumpy(vocab.vectors.data)
    else:
        return vocab.vectors.data
    
def get_features(texts, max_length, nlp, batch_size=64):
    X = np.zeros((len(texts), max_length), dtype="int32")
    i = 0
    for doc in nlp.pipe(texts, batch_size=batch_size, disable=["parser", "tagger", "ner"]):
        for j, token in enumerate(doc):
            if j >= max_length:
                break
            vector_id = token.vocab.vectors.find(key=token.orth)
            if vector_id >= 0:
                X[i, j] = vector_id
            else:
                X[i, j] = 0
        i += 1
    return X
    
def encode_labels(labels, one_hot=True):
    labels = labels.reset_index(drop=True)
    categories = list(labels.unique())
    Y = np.zeros((len(labels), 1))
    
    for i, cat in enumerate(categories):
        Y[labels.index[labels == cat]] = i
    
    if one_hot:
        return to_categorical(Y, num_classes=len(categories))
    else:
        return Y
        
def group_sentences(document, num_sentences=4):
    sentence_count = 0
    for token in document:
        if sentence_count % num_sentences == 0 and token.is_sent_start:
            token.is_sent_start = True
            sentence_count += 1
        elif token.is_sent_start:
            token.is_sent_start = False
            sentence_count += 1
    return document
