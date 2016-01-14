#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Examples of using DECO model.

"""
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd

from dcc import DCC


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

    model = DCC()
    ret, rho_series = model.simulate(nobs=nobs, ndim=ndim, volmean=volmean,
                                     persistence=persistence, beta=beta,
                                     acorr=acorr, bcorr=bcorr, rho=rho)

    ret.plot(subplots=True, sharey='row')
    plt.show()

    model = DCC(data=ret)

    result = model.fit(method='Nelder-Mead')

    model.std_data.plot(subplots=True, sharey='row')
    plt.show()
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

    print(model.param)
