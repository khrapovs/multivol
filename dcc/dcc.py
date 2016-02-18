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
from .data_dcc import DataDCC
from .arch_forecast import garch_forecast

__all__ = ['DCC']


class DCC(object):

    """DCC model.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, ret=None):
        """Initialize the model.

        """
        self.param = None
        self.data = DataDCC(ret=ret)

    @staticmethod
    def simulate(nobs=2000, ndim=3, persistence=.99, beta=.85,
                 volmean=.2, acorr=.15, bcorr=.8, rho=.9, error=None):
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
        if error is None:
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
        forecast = []
        theta = []
        data = self.data.ret.copy()
        for col in data:
            model = arch_model(data[col], p=1, q=1, mean='Zero',
                               vol='GARCH', dist='Normal')
            res = model.fit(disp='off')
            theta.append(res.params)
            vol.append(res.conditional_volatility)
            forecast.append(garch_forecast(res).iloc[-1, 0])
        theta = pd.concat(theta, axis=1)
        theta.columns = data.columns
        self.data.univ_vol = pd.concat(vol, axis=1)
        self.data.univ_vol.columns = data.columns
        self.param.univ = theta
        self.data.univ_forecast = np.array(forecast)

    def standardize_returns(self):
        """Standardize returns using estimated conditional volatility.

        """
        self.estimate_univ()
        self.data.std_ret = self.data.ret / self.data.univ_vol

    def filter_corr_dcc(self):
        """Filter DCC correlation matrix series.

        """
        data = self.data.std_ret.values
        neg_data = data.copy()
        neg_data[neg_data > 0] = 0
        nobs, ndim = data.shape
        acorr = self.param.acorr
        bcorr = self.param.bcorr
        dcorr = self.param.dcorr
        self.data.corr_dcc = np.zeros((nobs, ndim, ndim))
        self.data.qmat = np.zeros((nobs, ndim, ndim))
        self.data.qmat[0] = self.param.corr_target.copy()

        const = self.param.corr_target * (1 - acorr - bcorr) \
            - dcorr * self.param.corr_neg_target
        # Stationarity condition
        if not (scl.eigvals(const).real > 0).all():
            raise ValueError('Constant not positive definite!')

        for t in range(nobs):

            if t > 0:
                self.data.qmat[t] = const \
                    + acorr * data[t-1][:, np.newaxis] * data[t-1] \
                    + bcorr * self.data.qmat[t-1] \
                    + dcorr * neg_data[t-1][:, np.newaxis] * neg_data[t-1]

            qdiag = np.diag(self.data.qmat[t])
            if not (np.isfinite(qdiag).all() & (qdiag > 0).all()):
                raise ValueError('Invalid diagonal of Q matrix!')
            qdiag = qdiag**.5

            self.data.corr_dcc[t] = self.data.qmat[t] \
                 / (qdiag[:, np.newaxis] * qdiag)
            self.data.corr_dcc[t][np.diag_indices(ndim)] = np.ones(ndim)
            cond1 = (self.data.corr_dcc[t] >= -1).all()
            cond2 = (self.data.corr_dcc[t] <= 1).all()
            cond3 = np.isfinite(self.data.corr_dcc[t]).all()
            if not (cond1 & cond2 & cond3):
                raise ValueError('Invalid correlation matrix!')

    def forecast_qmat(self):
        """Forecast Q matrix.

        Returns
        -------
        (ndim, ndim) array

        """
        acorr = self.param.acorr
        bcorr = self.param.bcorr
        dcorr = self.param.dcorr
        data = self.data.std_ret.values[-1]
        neg_data = data.copy()
        neg_data[neg_data > 0] = 0
        return self.param.corr_target * (1 - acorr - bcorr) \
            - dcorr * self.param.corr_neg_target \
            + acorr * data[:, np.newaxis] * data \
            + bcorr * self.data.qmat[-1] \
            + dcorr * neg_data[:, np.newaxis] * neg_data

    def forecast_corr_dcc(self):
        """Forecast R matrix.

        Returns
        -------
        (ndim, ndim) array

        """
        qmat = self.forecast_qmat()
        qdiag = np.diag(qmat) ** .5
        return qmat / (qdiag[:, np.newaxis] * qdiag)

    def forecast_hmat(self):
        """Forecast H matrix.

        Returns
        -------
        (ndim, ndim) array

        """
        corr = self.forecast_corr_dcc()
        dmat = np.diag(self.data.univ_forecast)
        return dmat.dot(corr).dot(dmat)

    def filter_rho_series(self):
        """Filter rho series.

        """
        ndim = self.data.ndim
        self.data.rho_series = self.data.corr_dcc.sum((1, 2))
        self.data.rho_series = (self.data.rho_series / ndim - 1) / (ndim - 1)

    def corr_deco(self):
        """Construct DECO correlation matrix series.

        """
        nobs, ndim = self.data.nobs, self.data.ndim
        corr = np.zeros((nobs, ndim, ndim))
        for t in range(nobs):
            corr[t] = (1 - self.data.rho_series[t]) * np.eye(ndim) \
                    + self.data.rho_series[t] * np.ones((ndim, ndim))
        return corr

    def likelihood_value(self):
        """Log-likelihood function (data).

        """
        data = self.data.std_ret
        rho_series = self.data.rho_series
        nobs, ndim = data.shape
        corr_det = (1 - rho_series) ** (ndim - 1) \
            * (1 + (ndim - 1) * rho_series)
        out = np.log(corr_det) \
            + ((data**2).sum(1) - rho_series * data.sum(1)**2 \
            / (1 + (ndim - 1) * rho_series)) / (1 - rho_series)
        return np.mean(out)

    def likelihood(self, theta):
        """Log-likelihood function (parameters).

        """
        # Stationarity conditions
        if (theta[:2] <= 0).any():
            return 1e10
        if (theta[:2].sum() + theta[2] * self.lmbd) >= 1:
            return 1e10

        self.param.update_dcc(theta)

        try:
            self.filter_corr_dcc()
        except ValueError:
            return 1e10

        self.filter_rho_series()

        return self.likelihood_value()

    def fit(self, theta_start=[.1, .5, 0.], method='SLSQP'):
        """Fit DECO model to the data.

        """
        self.param = ParamDCC(ndim=self.data.ndim)
        self.standardize_returns()
        self.param.corr_target = np.corrcoef(self.data.std_ret.T)
        neg_ret = self.data.std_ret.T.copy()
        neg_ret[neg_ret > 0] = 0
        self.param.corr_neg_target = np.corrcoef(neg_ret)

        # Compute lmbd to check for stationarity of Q
        factor, lower = scl.cho_factor(self.param.corr_target, lower=True)
        sandwich = scl.solve_triangular(factor, self.param.corr_neg_target,
                                        lower=lower)
        sandwich = scl.solve_triangular(factor, sandwich.T,
                                        lower=lower)
        self.lmbd = scl.eigvals(sandwich).real.max()

        options = {'disp': False, 'maxiter': int(1e6)}

        opt_out = sco.minimize(self.likelihood, theta_start,
                               method=method, options=options)

        self.param.abcorr = opt_out.x
        self.data.rho_series = pd.Series(self.data.rho_series,
                                         index=self.data.ret.index)
        self.estimate_innov()
        return opt_out

    def estimate_innov(self):
        """Estimate multivariate innovations.

        """
        nobs, ndim = self.data.ret.shape
        innov = np.zeros((nobs, ndim))
        data = self.data.std_ret.values
        for t in range(nobs):
            try:
                factor, lower = scl.cho_factor(self.data.corr_dcc[t],
                                               lower=True)
            except:
                print('Cholesky failed!', t)
                print(self.data.corr_dcc[t])
                factor, lower = np.eye(ndim), True
            innov[t] = scl.solve_triangular(factor, data[t], lower=lower)

        self.data.innov = pd.DataFrame(innov, index=self.data.ret.index,
                                       columns=self.data.ret.columns)
