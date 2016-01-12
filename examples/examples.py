#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Examples of using ARCH package.

"""
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.linalg as scl


if __name__ == '__main__':

    pd.set_option('float_format', '{:6.4f}'.format)
    np.set_printoptions(precision=4, suppress=True)
    sns.set()

#    persistence = .99
#    beta = persistence
#    gamma = .1
#    alpha = persistence - beta
#    omega = 0
#    eta, lam = 30, .0
#    nobs = 2000
#
#    model = arch_model(None, p=1, o=1, q=1, mean='Zero',
#                       vol='EGARCH', dist='Normal')
#
#    data = model.simulate([omega, alpha, gamma, beta], nobs=nobs)
#
#    data[['data', 'volatility']].plot()
#    plt.show()

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

    pd.DataFrame(ret).plot(subplots=True, sharey='row')
    plt.show()

    plt.plot(rho)
    plt.show()
