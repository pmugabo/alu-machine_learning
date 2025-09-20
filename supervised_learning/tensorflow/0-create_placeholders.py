#!/usr/bin/env python3

"""This module contains a function that
returns two placeholders for a
neural network"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """nx - no. of feauture columns in data
    classe - no. of classes in out classifier
    x - placeholder for input data
    y - placeholder for one-hot labels for input data
    """
    X = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    Y = tf.placeholder(tf.float32, shape=[None, classes], name='y')

    return X, Y
