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

    nobs = 1000
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
#    vol.plot(subplots=True, sharey='row')
#    plt.show()

    std_data = model.standardize_returns(ret)
    std_data.plot(subplots=True, sharey='row')
    plt.show()

    rho_series_fit, corr = model.filter_deco(data=std_data, param=param)

    rho_series.plot()
    plt.plot(rho_series_fit)
    plt.show()

    print(model.likelihood(data=ret, rho_series=rho_series_fit))
