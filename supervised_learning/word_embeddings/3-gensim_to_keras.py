#!/usr/bin/env python3
"""
Convert Gensim Word2Vec to Keras Embedding Layer
"""

import numpy as np
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.initializers import Constant


def gensim_to_keras(model):
    """
    Converts a trained Gensim Word2Vec
    model to a Keras Embedding layer.
    """
    # Get the Word2Vec vocabulary size
    # (number of words) and vector size (embedding dimensions)
    vocab_size = len(model.wv.key_to_index)
    vector_size = model.wv.vector_size

    # Extract the weights (word vectors) from the Gensim model
    weights = np.zeros((vocab_size, vector_size))
    for i, word in enumerate(model.wv.index_to_key):
        weights[i] = model.wv[word]

    # Create a Keras Embedding layer with the Gensim weights
    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        embeddings_initializer=Constant(weights),
        trainable=True
    )

    return embedding_layer
