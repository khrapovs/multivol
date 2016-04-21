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
from dcc.utils import take_time


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

    ret, rho_series = DCC.simulate(nobs=nobs, ndim=ndim, volmean=volmean,
                                     persistence=persistence, beta=beta,
                                     acorr=acorr, bcorr=bcorr, rho=rho)

    model = DCC(ret=ret)

    with take_time('Python'):
        result = model.fit(method='Nelder-Mead', numba=False)
    with take_time('Numba'):
        result = model.fit(method='Nelder-Mead', numba=True)

    print(model.data)

    model.data.plot_returns()
    model.data.plot_std_returns()
    model.data.plot_innov()

    print(result)
    print(model.param)

    model.data.rho_series.plot(label='Fitted')
    rho_series.plot(label='Simulated')
    plt.legend()
    plt.show()

    print(model.data.innov_corr())

    print('Volatility forecast:\n', model.forecast_hmat())
