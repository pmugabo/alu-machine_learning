#!/usr/bin/env python3
"""This module contains a function that perfoms
finds the best number of clusters for a GMM using the
Bayesian Information Criterion"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    finds the best number of clusters for a GMM using the
    Bayesian Information Criterion
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None

    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None

    if kmax is None:
        kmax = X.shape[0]

    if not isinstance(kmax, int) or kmax < kmin:
        return None, None, None, None

    if not isinstance(iterations, int):
        return None, None, None, None

    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    likelyhoods = []
    bics = []
    best_k = None
    best_res = None
    best_bic = None

    for k in range(kmin, kmax + 1):
        pi, m, S, g, ll = expectation_maximization(X, k, iterations, tol,
                                                     verbose)
        p = (k - 1) + k * d + k * d * (d + 1) / 2
        bic = p * np.log(n) - 2 * ll

        likelyhoods.append(ll)
        bics.append(bic)

        if best_bic is None or bic < best_bic:
            best_bic = bic
            best_k = k
            best_res = (pi, m, S)

    return best_k, best_res, np.asarray(likelyhoods), np.asarray(bics)