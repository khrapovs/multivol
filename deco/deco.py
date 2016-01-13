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
import scipy.optimize as sco

from arch import arch_model
from .param_deco import ParamDECO

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

        hvar = np.zeros((nobs+1, ndim, ndim))
        qmat = np.zeros((nobs+1, ndim, ndim))
        rho_series = np.ones(nobs+1)
        dvec = np.ones(ndim) * volmean
        qmat[0] = self.param.corr_target
        ret = np.zeros((nobs+1, ndim))
        mean, cov = np.zeros(ndim), np.eye(ndim)
        error = np.random.multivariate_normal(mean, cov, nobs+1)
        error = (error - error.mean(0)) / error.std(0)
        qeta = np.zeros(ndim)

        for t in range(1, nobs+1):
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

        return pd.DataFrame(ret[1:]), pd.Series(rho_series[1:])

    def estimate_univ(self, data=None):
        """Estimate univariate volatility models.

        """
        vol = []
        theta = []
        for ret in data.values.T:
            model = arch_model(ret, p=1, q=1, mean='Zero',
                               vol='GARCH', dist='Normal')
            res = model.fit(disp='off')
            theta.append(res.params)
            vol.append(res.conditional_volatility)
        theta = pd.concat(theta, axis=1)
        theta.columns = data.columns
        vol = pd.DataFrame(np.vstack(vol).T, columns=data.columns)
        return vol, theta

    def standardize_returns(self, data=None):
        """Standardize returns using estimated conditional volatility.

        """
        return data / self.estimate_univ(data=data)[0]

    def filter_deco(self, data=None, param=None):
        """Filter Q matrix series.

        """
        data = data.values
        nobs, ndim = data.shape
        qmat = np.zeros((nobs, ndim, ndim))
        rho_series = np.ones(nobs)

        acorr = param.acorr
        bcorr = param.bcorr

        for t in range(nobs):
            if t == 0:
                qmat[0] = self.param.corr_target
            else:
                qmat[t] = qmat[0] * (1 - acorr - bcorr) \
                    + acorr * data[t-1][:, np.newaxis] * data[t-1] \
                    + bcorr * qmat[t-1]
            qdiag = np.diag(qmat[t]) ** .5
            corr_dcc = (1 / qdiag[:, np.newaxis] / qdiag) * qmat[t]
            rho_series[t] = (corr_dcc.sum() - ndim) / (ndim - 1) / ndim
        return rho_series

    def corr_deco(self, data=None, rho_series=None):
        """Construct correlation matrix series.

        """
        nobs, ndim = data.shape
        corr_deco = np.zeros((nobs, ndim, ndim))
        for t in range(nobs):
            corr_deco[t] = (1 - rho_series[t]) * np.eye(ndim) \
                    + rho_series[t] * np.ones((ndim, ndim))
        return corr_deco

    def likelihood_value(self, rho_series=None):
        """Log-likelihood function (data).

        """
        # TODO: Should be done outside!
        data = self.data
        nobs, ndim = data.shape
        out = np.log((1 - rho_series) ** (ndim - 1) \
            * (1 + (ndim - 1) * rho_series)) \
            + ((data**2).sum(1) - rho_series * data.sum(1)**2 \
            / (1 + (ndim - 1) * rho_series)) / (1 - rho_series)
        return np.mean(out)

    def likelihood(self, theta):
        """Log-likelihood function (parameters).

        """
        self.param.update_deco(theta)
        if (np.sum(theta) >= 1.) or (theta <= 0.).any():
            return 1e10
        else:
            rho_series = self.filter_deco(data=self.std_data,
                                          param=self.param)
            self.rho_series = pd.Series(rho_series, index=self.data.index)
            return self.likelihood_value(rho_series=rho_series)

    def fit(self, theta_start=[.1, .5], data=None, method='SLSQP'):
        """Fit DECO model to the data.

        """
        self.data = data
        self.std_data = self.standardize_returns(self.data)
        self.param.corr_target = np.corrcoef(self.std_data.T)
        # Optimization options
        options = {'disp': False, 'maxiter': int(1e6)}
        opt_out = sco.minimize(self.likelihood, theta_start,
                               method=method, options=options)
        return opt_out
