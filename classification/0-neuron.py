#!/usr/bin/env python3
""" Module to implement Neuron class
"""
import numpy as np


class Neuron:
    """ Class that implements a neuron that performs
    binary classification
    """
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')

        if nx < 1:
            raise ValueError('nx must be a positive integer')

        self.W = np.random.normal(0, 1, (1, nx))
        self.b = 0
        self.A = 0