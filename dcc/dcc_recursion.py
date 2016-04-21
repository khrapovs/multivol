#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DCC model
=========

"""
from __future__ import print_function, division

import numpy as np

from numba import jit, float64, void

__all__ = ['dcc_recursion_python', 'dcc_recursion_numba',
           'corr_dcc_python','corr_dcc_numba']


def dcc_recursion_python(qmat, const, data, neg_data, param):
    """DCC recursion.

    Parameters
    ----------
    qmat : (nobs, ndim, ndim) array
        Raw correlation matrix
    const : (ndim, ndim) array
        Constant
    data : (nobs, ndim) array
        Innovations
    neg_data : (nobs, ndim) array
        Only negative innovations
    param : (3,) array
        DCC paeameters

    """

    acorr, bcorr, dcorr = param
    nobs = data.shape[0]

    for t in range(1, nobs):
        qmat[t] = const \
            + acorr * data[t-1][:, np.newaxis] * data[t-1] \
            + bcorr * qmat[t-1] \
            + dcorr * neg_data[t-1][:, np.newaxis] * neg_data[t-1]


@jit(void(float64[:, :, :], float64[:, :], float64[:, :],
          float64[:, :], float64[:]), nopython=True, nogil=True, cache=True)
def dcc_recursion_numba(qmat, const, data, neg_data, param):
    """DCC recursion.

    Parameters
    ----------
    qmat : (nobs, ndim, ndim) array
        Raw correlation matrix
    const : (ndim, ndim) array
        Constant
    data : (nobs, ndim) array
        Innovations
    neg_data : (nobs, ndim) array
        Only negative innovations
    param : (3,) array
        DCC paeameters

    """

    acorr, bcorr, dcorr = param
    nobs, ndim = data.shape

    for t in range(1, nobs):
        for i in range(ndim):
            for j in range(ndim):
                qmat[t, i, j] = const[i, j] \
                    + acorr * data[t-1, i] * data[t-1, j] \
                    + bcorr * qmat[t-1, i, j] \
                    + dcorr * neg_data[t-1, i] * neg_data[t-1, j]


def corr_dcc_python(corr_dcc, qmat):
    """Convert Q matrix to correlation matrix.

    Parameters
    ----------
    corr_dcc : (nobs, ndim, ndim) array
        Correlation matrix
    qmat : (nobs, ndim, ndim) array
        Raw correlation matrix

    """
    nobs, ndim = qmat.shape[:2]

    for t in range(nobs):

        qdiag = np.diag(qmat[t])**.5
#        if not (np.isfinite(qdiag).all() & (qdiag > 0).all()):
#            raise ValueError('Invalid diagonal of Q matrix!')
#        qdiag = qdiag**.5

        corr_dcc[t] = qmat[t] / (qdiag[:, np.newaxis] * qdiag)
        corr_dcc[t][np.diag_indices(ndim)] = np.ones(ndim)


@jit(void(float64[:, :, :], float64[:, :, :]),
          nopython=True, nogil=True, cache=True)
def corr_dcc_numba(corr_dcc, qmat):
    """Convert Q matrix to correlation matrix.

    Parameters
    ----------
    corr_dcc : (nobs, ndim, ndim) array
        Correlation matrix
    qmat : (nobs, ndim, ndim) array
        Raw correlation matrix

    """
    nobs, ndim = qmat.shape[:2]

    for t in range(nobs):

        for i in range(ndim):
            for j in range(ndim):

                corr_dcc[t, i, j] = qmat[t, i, j] \
                    / (qmat[t, i, i] * qmat[t, j, j])**.5

            corr_dcc[t, i, i] = 1.
