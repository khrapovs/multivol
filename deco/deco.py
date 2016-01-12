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

from arch import arch_model
from deco import ParamDECO

__all__ = ['DECO']


class DECO(object):

    """DECO model.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, param=ParamDECO()):
        """Initialize the model.

        """
        self.param = param

    def simulate(self, nobs=2000):
        """Simulate returns and (co)variances.

        Parameters
        ----------

        Returns
        -------

        """
        ndim = self.param.ndim
        persistence = self.param.persistence
        beta = self.param.beta
        alpha = self.param.alpha
        volmean = self.param.volmean

        bcorr = self.param.bcorr
        acorr = self.param.acorr
        rho = self.param.rho

        hvar = np.zeros((nobs, ndim, ndim))
        qmat = np.zeros((nobs, ndim, ndim))
        rho_series = np.ones(nobs) * rho
        dvec = np.ones(ndim) * volmean
        qmat[0] = (1 - rho) * np.eye(ndim) + rho * np.ones((ndim, ndim))
        ret = np.zeros((nobs, ndim))
        mean, cov = np.zeros(ndim), np.eye(ndim)
        error = np.random.multivariate_normal(mean, cov, nobs)
        error = (error - error.mean(0)) / error.std(0)
        qeta = np.zeros(ndim)

        for t in range(1, nobs):
            dvec = volmean * (1 - persistence) \
                + alpha * ret[t-1]**2 + beta * dvec
            qmat[t] = qmat[0] * (1 - acorr - bcorr) \
                + acorr * qeta[:, np.newaxis] * qeta \
                + bcorr * qmat[t-1]
            qdiag = np.diag(qmat[t]) ** .5
            corr_dcc = (1 / qdiag[:, np.newaxis] / qdiag) * qmat[t]
            rho_series[t] = (corr_dcc.sum() - ndim) / (ndim - 1) / ndim
            corr = (1 - rho_series[t]) * np.eye(ndim) \
                + rho_series[t] * np.ones((ndim, ndim))
            hvar[t] = (dvec[:, np.newaxis] * dvec)**.5 * corr
            ret[t] = (error[t] * scl.cholesky(hvar[t], 1)).sum(1)
            qeta = qdiag * ret[t] / dvec**.5

        return pd.DataFrame(ret), pd.Series(rho_series)

    def estimate_univ(self, data=None):
        """Estimate univariate volatility models.

        """
        vol = []
        theta = []
        for ret in data.T.values:
            model = arch_model(ret, p=1, q=1, mean='Zero',
                               vol='GARCH', dist='Normal')
            res = model.fit(disp='off')
            theta.append(res.params)
            vol.append(res.conditional_volatility)
        theta = pd.concat(theta, axis=1)
        theta.columns = data.columns
        vol = np.vstack(vol)
        return vol, theta
