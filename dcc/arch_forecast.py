#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ARCH forecast
=============

The code is found here: https://github.com/bashtage/arch/issues/38
The author is https://github.com/Vincent-Chin

"""
from __future__ import print_function, division

import re
import pandas as pd
import numpy as np


def rForecastVar(omega, factors, multipliers, h):
    """Recursive calculator for garch forecasting.
    Generates variance, not volatility.

    """
    if h == 0:
        raise "Invalid horizon.  Was less than 1."
    if h == 1:
        return omega + np.sum(factors * multipliers, axis=1)
    else:
        return omega + (np.sum(factors, axis=1)
                          * rForecastVar(omega, factors, multipliers, h - 1))


def garch_forecast(archmodelresult, horizon=1):
    """This method manually forecasts the arch-family of models, since
    it is not yet implemented in bashtage/arch

    """
    model_params = archmodelresult.params
    resids = archmodelresult.resid # for ZARCH-style models that depend on sign
    resid2 = archmodelresult.resid ** 2
    convol2 = archmodelresult.conditional_volatility ** 2

    # first generate the derived datasets
    factors = pd.DataFrame(index=convol2.index)
    forecasts = pd.DataFrame(index=convol2.index)
    multipliers = pd.DataFrame(index=convol2.index)

    for key in model_params.keys():
        # get the offset (for alpha and beta)
        offset = re.findall(r'\d+', key)
        if len(offset) == 1:
            offset = int(offset[0]) - 1
        else:
            offset = 0

        # generate a value column for this key into the response
        if 'mu' in key:
            pass
        if 'omega' in key:
            pass
        if 'alpha' in key:
            factors[key] = model_params[key]
            multipliers[key] = resid2.shift(offset)
        if 'gamma' in key:
            factors[key] = model_params[key]
            multipliers[key] = resid2.shift(offset)
            multipliers[key][resids >= 0] = 0
        if 'beta' in key:
            factors[key] = model_params[key]
            multipliers[key] = convol2.shift(offset)

    # note: these forecasts are aligned such that the forecast is for the
    # 'next' row of returns data.
    omega = model_params['omega']
    for h in range(1, horizon + 1):
        colName = 'foVar[' + str(h) + ']'
        forecasts[colName] = rForecastVar(omega, factors, multipliers, h)

    # sums are invalid when anything is nan
    nonValues = np.isnan(factors).any(axis=1)
    for idx in nonValues[nonValues == True].index:
        forecasts.ix[idx,] = np.nan

    # return the conditional volatility, not conditional variance
    forecasts = np.sqrt(forecasts)

    return forecasts
