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

    model = DCC(data=ret)

    model.standardize_returns()
    model.std_data.plot(subplots=True, sharey='row')
    plt.show()

    result = model.fit(method='Nelder-Mead')
    print(result)
    print(model.param)

    model.rho_series.plot(label='Fitted')
    rho_series.plot(label='Simulated')
    plt.legend()
    plt.show()

    model.estimate_innov()
    model.innov.plot(subplots=True, sharey='row')
    plt.show()

    print(np.corrcoef(model.innov.T))
