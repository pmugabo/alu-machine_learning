#!/usr/bin/env python3
"""
Word2Vec Model Creation
"""

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5, cbow=True, iterations=5, seed=0, workers=1):
    """
    Word 2 Vec function
    """
    # Set the training algorithm to CBOW or Skip-gram
    sg = 0 if cbow else 1

    # Initialize and train the Word2Vec model
    model = Word2Vec(
        sentences=sentences,
        vector_size=size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,  # 0 for CBOW, 1 for Skip-gram
        negative=negative,
        seed=seed,
        epochs=iterations
    )

    return model
