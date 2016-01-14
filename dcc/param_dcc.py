#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DCC model parameters
====================

"""
from __future__ import print_function, division

import numpy as np
import pandas as pd

__all__ = ['ParamDCC']


class ParamDCC(object):

    """DCC model parameters.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, ndim=None, univ=None, abcorr=None):
        """Initialize parameter class.

        """
        self.ndim = ndim
        self.univ = univ
        self.abcorr = abcorr
        self.corr_target = None

    def as_pandas(self):
        """Represent parameters as pandas objects.

        """
        corr = pd.Series({'Feedback': self.abcorr[1],
                          'Persistence': self.abcorr.sum()})
        return corr

    def update_dcc(self, theta=None):
        """Update DCC parameters.

        """
        self.acorr, self.bcorr = theta

    def __str__(self):
        """String representation.

        """
        corr = self.as_pandas()
        width = 60
        show = '=' * width
        show += '\nNumber of dimensions = %d' % self.ndim
        show += '\n\nParameters of univariate volatilities:\n'
        show += self.univ.to_string(float_format=lambda x: '%.2f' % x)
        show += '\n\nParameters of correlation model:\n'
        show += corr.to_string(float_format=lambda x: '%.2f' % x)
        show += '\nCorrelation target:\n' + np.array_str(self.corr_target)
        show += '\n' + '=' * width
        return show + '\n'

    def __repr__(self):
        """String representation.

        """
        return self.__str__()
