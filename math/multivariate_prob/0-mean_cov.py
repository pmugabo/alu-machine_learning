#!/usr/bin/env python3
""" Implements mean_cov
"""
import numpy as np


def mean_cov(X):
    """ Calculates the mean and covariance of a matrix
    Args:
        X (numpy.ndarray) : A 2D array representing the dataset
    Returns:
        mean (numpy.ndarray): The mean vector of the dataset
        cov (numpy.ndarray): A 2D array representing the covariance
        of the dataset
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError('X must be a 2D numpy.ndarray')

    n, _ = X.shape

    if n < 2:
        raise ValueError('X must contain multiple data points')

    mean = np.mean(X, axis=0).reshape(1, -1)
    deviations = X - mean
    cov = (deviations.T @ deviations) / (n - 1)

    return mean, cov
