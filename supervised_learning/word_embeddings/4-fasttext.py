#!/usr/bin/env python3
"""
FastText Model Creation
"""

from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5, cbow=True, iterations=5, seed=0, workers=1):
    """
    Creates and trains a Gensim FastText model.
    """
    # Set the training algorithm to CBOW or Skip-gram
    sg = 0 if cbow else 1

    # Initialize and train the FastText model
    model = FastText(
        sentences=sentences,
        vector_size=size,    # Size of the word vectors
        window=window,       # Context window size
        min_count=min_count, # Minimum word count threshold
        sg=sg,               # 0 for CBOW, 1 for Skip-gram
        negative=negative,   # Negative sampling
        seed=seed,           # Seed for reproducibility
        epochs=iterations,   # Number of epochs (iterations)
        workers=workers      # Number of worker threads
    )

    return model
