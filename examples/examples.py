#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Examples of using DECO model.

"""
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd

from dcc import DCC, ParamDCC


if __name__ == '__main__':

    pd.set_option('float_format', '{:6.4f}'.format)
    np.set_printoptions(precision=4, suppress=True)
    sns.set()

    nobs = 500
    ndim = 3
    persistence = .99
    beta = .85
    volmean = .2

    acorr = .05
    bcorr = .9
    rho = .5

    param = ParamDCC(ndim=ndim, persistence=persistence, beta=beta,
                     volmean=volmean, acorr=acorr, bcorr=bcorr, rho=rho)

    print(param)

    model = DCC(param)
    ret, rho_series = model.simulate(nobs=nobs)

    ret.plot(subplots=True, sharey='row')
    plt.show()
    rho_series.plot()
    plt.show()

#    vol, theta = model.estimate_univ(ret)
#    vol.plot(subplots=True, sharey='row')
#    plt.show()

    std_data = model.standardize_returns(ret)
    std_data.plot(subplots=True, sharey='row')
    plt.show()

    result = model.fit(data=ret, method='Nelder-Mead')
    print(result)
    print(model.param)

    model.rho_series.plot(label='Fitted')
    rho_series.plot(label='Simulated')
    plt.legend()
    plt.show()

    model.estimate_residuals()
    model.errors.plot(subplots=True, sharey='row')
    plt.show()

    print(np.corrcoef(model.errors.T))
