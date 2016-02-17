#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DCC model data
==============

"""
from __future__ import print_function, division

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

__all__ = ['DataDCC']


class DataDCC(object):

    """DCC model data.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, ret=None):
        """Initialize parameter class.

        """
        self.ret = ret
        self.nobs = None
        self.ndim = None
        self.std_ret = None
        self.innov = None
        self.rho_series = None
        self.qmat = None
        self.corr_dcc = None
        self.univ_forecast = None
        if ret is not None:
            self.nobs, self.ndim = ret.shape

    def __str__(self):
        """String representation.

        """
        descr = self.ret.describe()
        width = 60
        show = '=' * width
        show += '\nNumber of returns = %d\n' % self.ndim
        show += descr.to_string(float_format=lambda x: '%.2f' % x)
        show += '\n' + '=' * width
        return show + '\n'

    def __repr__(self):
        """String representation.

        """
        return self.__str__()

    def plot_returns(self):
        """Plot raw returns.

        """
        self.ret.plot(subplots=True, sharey='row')
        plt.show()

    def plot_std_returns(self):
        """Plot standardized returns.

        """
        self.std_ret.plot(subplots=True, sharey='row')
        plt.show()

    def plot_innov(self):
        """Plot innovations.

        """
        self.innov.plot(subplots=True, sharey='row')
        plt.show()

    def innov_corr(self):
        """Innovation correlation.

        """
        return np.corrcoef(self.innov.T)
