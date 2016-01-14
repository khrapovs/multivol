#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DCC model
=========

"""
from __future__ import print_function, division

import numpy as np
import pandas as pd
import scipy.linalg as scl
import scipy.optimize as sco

from arch import arch_model
from .param_dcc import ParamDCC

__all__ = ['DCC']


class DCC(object):

    """DECO model.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, data=None):
        """Initialize the model.

        """
        self.param = None
        self.data = data

    def simulate(self, nobs=2000, ndim=3, persistence=.99, beta=.85,
                 volmean=.2, acorr=.15, bcorr=.8, rho=.9):
        """Simulate returns and (co)variances.

        Parameters
        ----------

        Returns
        -------

        """
        alpha = persistence - beta
        hvar = np.zeros((nobs+1, ndim, ndim))
        rho_series = np.ones(nobs+1)
        dvec = np.ones(ndim) * volmean
        corr_target = (1 - rho) * np.eye(ndim) \
            + rho * np.ones((ndim, ndim))
        qmat = corr_target.copy()
        ret = np.zeros((nobs+1, ndim))
        mean, cov = np.zeros(ndim), np.eye(ndim)
        error = np.random.multivariate_normal(mean, cov, nobs+1)
        error = (error - error.mean(0)) / error.std(0)
        qeta = np.zeros(ndim)

        for t in range(1, nobs+1):
            dvec = volmean * (1 - persistence) \
                + alpha * ret[t-1]**2 + beta * dvec
            qmat = corr_target * (1 - acorr - bcorr) \
                + acorr * qeta[:, np.newaxis] * qeta \
                + bcorr * qmat
            qdiag = np.diag(qmat) ** .5
            corr_dcc = (1 / qdiag[:, np.newaxis] / qdiag) * qmat
            rho_series[t] = (corr_dcc.sum() - ndim) / (ndim - 1) / ndim
            hvar[t] = (dvec[:, np.newaxis] * dvec)**.5 * corr_dcc
            ret[t] = error[t].dot(scl.cholesky(hvar[t], 0))
            qeta = qdiag * ret[t] / dvec**.5

        return pd.DataFrame(ret[1:]), pd.Series(rho_series[1:])

    def estimate_univ(self):
        """Estimate univariate volatility models.

        """
        vol = []
        theta = []
        data = self.data.copy()
        for ret in data.values.T:
            model = arch_model(ret, p=1, q=1, mean='Zero',
                               vol='GARCH', dist='Normal')
            res = model.fit(disp='off')
            theta.append(res.params)
            vol.append(res.conditional_volatility)
        theta = pd.concat(theta, axis=1)
        theta.columns = data.columns
        self.univ_vol = pd.DataFrame(np.vstack(vol).T, index=data.index,
                                     columns=data.columns)
        self.param.univ = theta

    def standardize_returns(self):
        """Standardize returns using estimated conditional volatility.

        """
        self.estimate_univ()
        self.std_data = self.data / self.univ_vol

    def filter_corr_dcc(self):
        """Filter DCC correlation matrix series.

        """
        data = self.std_data.values
        nobs, ndim = data.shape
        acorr = self.param.acorr
        bcorr = self.param.bcorr
        self.corr_dcc = np.zeros((nobs, ndim, ndim))
        qmat = self.param.corr_target.copy()

        for t in range(nobs):
            if t > 0:
                qmat = self.param.corr_target * (1 - acorr - bcorr) \
                    + acorr * data[t-1][:, np.newaxis] * data[t-1] \
                    + bcorr * qmat
            qdiag = np.diag(qmat) ** .5
            self.corr_dcc[t] = (1 / qdiag[:, np.newaxis] / qdiag) * qmat

    def filter_rho_series(self):
        """Filter rho series.

        """
        nobs, ndim = self.data.shape
        self.rho_series = np.array([(corr.sum() - ndim) / (ndim - 1) / ndim
            for corr in self.corr_dcc])

    def corr_deco(self):
        """Construct DECO correlation matrix series.

        """
        nobs, ndim = self.data.shape
        corr = np.zeros((nobs, ndim, ndim))
        for t in range(nobs):
            corr[t] = (1 - self.rho_series[t]) * np.eye(ndim) \
                    + self.rho_series[t] * np.ones((ndim, ndim))
        return corr

    def likelihood_value(self):
        """Log-likelihood function (data).

        """
        data = self.std_data
        nobs, ndim = data.shape
        corr_det = (1 - self.rho_series) ** (ndim - 1) \
            * (1 + (ndim - 1) * self.rho_series)
        out = np.log(corr_det) \
            + ((data**2).sum(1) - self.rho_series * data.sum(1)**2 \
            / (1 + (ndim - 1) * self.rho_series)) / (1 - self.rho_series)
        return np.mean(out)

    def likelihood(self, theta):
        """Log-likelihood function (parameters).

        """
        self.param.update_dcc(theta)
        if (np.sum(theta) >= 1.) or (theta <= 0.).any():
            return 1e10
        else:
            self.filter_corr_dcc()
            self.filter_rho_series()
            return self.likelihood_value()

    def fit(self, theta_start=[.1, .5], method='SLSQP'):
        """Fit DECO model to the data.

        """
        self.param = ParamDCC(ndim=self.data.shape[1])
        self.standardize_returns()
        self.param.corr_target = np.corrcoef(self.std_data.T)
        options = {'disp': False, 'maxiter': int(1e6)}
        opt_out = sco.minimize(self.likelihood, theta_start,
                               method=method, options=options)
        self.param.acorr, self.param.bcorr = opt_out.x
        self.rho_series = pd.Series(self.rho_series, index=self.data.index)
        return opt_out

    def estimate_innov(self):
        """Estimate multivariate innovations.

        """
        nobs, ndim = self.data.shape
        innov = np.zeros((nobs, ndim))
        data = self.std_data.values
        for t in range(nobs):
            factor, lower = scl.cho_factor(self.corr_dcc[t], lower=True)
            innov[t] = scl.solve_triangular(factor, data[t], lower=lower)
        self.innov = pd.DataFrame(innov, index=self.data.index,
                                  columns=self.data.columns)
