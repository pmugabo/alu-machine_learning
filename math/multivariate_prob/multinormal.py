#!/usr/bin/env python3

"""
This module contains a class that represents
a Multivariate Normal distribution.
"""
import numpy as np


class MultiNormal:
    """
    class initialization
    """

    def __init__(self, data):
        """
        class constructor
        data - numpy.ndarray - shape (d, n)-
        containing the data set:
        n - int - number of data points
        d - int - number of dimensions in each data point
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')

        if data.shape[1] < 2:
            raise ValueError('data must contain multiple data points')

        mean = np.mean(data, axis=1).reshape(data.shape[0], 1)
        self.mean = mean
        cov = np.matmul(data - mean, (data - mean).T) / (data.shape[1] - 1)
        self.cov = cov

    def pdf(self, x):
        """
        calculate PDF at a data point
        x - numpy.ndarray - shape (d, 1) - containing the data point
        d - int - number of dimensions of the Multinomial instance
        """

        if not isinstance(x, np.ndarray):
            raise TypeError('x must be a numpy.ndarray')

        if x.shape != (self.mean.shape[0], 1):
            raise ValueError(
                'x must have the shape ({:d}, 1)'.
                format(self.mean.shape[0])
            )

        # calculate PDF
        d = self.cov.shape[0]
        x_m = x - self.mean
        cov_inv = np.linalg.inv(self.cov)
        det = np.linalg.det(self.cov)
        prefactor = 1.0 / (np.sqrt((2 * np.pi) ** d * det))
        exponent = -0.5 * np.matmul(np.matmul(x_m.T, cov_inv), x_m)
        pdf = prefactor * np.exp(exponent)
        return pdf[0][0]
