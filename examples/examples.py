#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Examples of using DECO model.

"""
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd

from deco import DECO, ParamDECO


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
    volmean = .2

    bcorr = .8
    acorr = .15
    rho = .9

    param = ParamDECO(ndim=ndim, persistence=persistence, beta=beta,
                      volmean=volmean, acorr=acorr, bcorr=bcorr, rho=rho)

    print(param)

    model = DECO(param)
    ret, rho_series = model.simulate(nobs=nobs)

    ret.plot(subplots=True, sharey='row')
    plt.show()

    rho_series.plot()
    plt.show()

#    vol, theta = model.estimate_univ(ret)
    std_data = model.standardize_returns(ret)

    std_data.plot(subplots=True, sharey='row')
    plt.show()
