#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DECO model
==========

"""
from __future__ import print_function, division

import numpy as np
import pandas as pd
import scipy.linalg as scl


class DECO(object):

    """DECO model.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self):
        pass

    def simulate(self, param=None, nobs=1000, ):
        """Simulate returns and (co)variances.

        Parameters
        ----------

        Returns
        -------

        """
        nobs = 2000
        ndim = 3
        persistence = .99
        beta = .85
        alpha = persistence - beta
        volmean = .2
        omega = volmean * (1 - persistence)

        bcorr = .8
        acorr = .15
        prho = .9

        hvar = np.zeros((nobs, ndim, ndim))
        qmat = np.zeros((nobs, ndim, ndim))
        rho = np.ones(nobs) * prho
        dvec = np.ones(ndim) * volmean
        qmat[0] = (1 - prho) * np.eye(ndim) + prho * np.ones((ndim, ndim))
        ret = np.zeros((nobs, ndim))
        mean, cov = np.zeros(ndim), np.eye(ndim)
        error = np.random.multivariate_normal(mean, cov, nobs)
        error = (error - error.mean(0)) / error.std(0)
        qeta = np.zeros(ndim)

        for t in range(1, nobs):
            dvec = omega + alpha * ret[t-1]**2 + beta * dvec
            qmat[t] = qmat[0] * (1 - acorr - bcorr) \
                + acorr * qeta[:, np.newaxis] * qeta \
                + bcorr * qmat[t-1]
            qdiag = np.diag(qmat[t]) ** .5
            corr_dcc = (1 / qdiag[:, np.newaxis] / qdiag) * qmat[t]
            rho[t] = (corr_dcc.sum() - ndim) / (ndim - 1) / ndim
            corr = (1 - rho[t]) * np.eye(ndim) + rho[t] * np.ones((ndim, ndim))
            hvar[t] = (dvec[:, np.newaxis] * dvec)**.5 * corr
            ret[t] = (error[t] * scl.cholesky(hvar[t], 1)).sum(1)
            qeta = qdiag * ret[t] / dvec**.5

        return pd.DataFrame(ret), pd.Series(rho)
