#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DECO model
==========

"""
from __future__ import print_function, division

import numpy as np
import pandas as pd
import scipy.linalg as scl

__all__ = ['ParamDECO']


class ParamDECO(object):

    """DECO model parameters.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, ndim=3, persistence=.99, beta=.85, volmean=.2,
                 acorr=.15, bcorr=.8, rho=.9):
        """Initialize parameter class.

        """
        self.ndim = ndim
        self.persistence = persistence * np.ones(ndim)
        self.beta = beta * np.ones(ndim)
        self.alpha = self.persistence - self.beta
        self.volmean = volmean * np.ones(ndim)

        self.acorr = acorr
        self.bcorr = bcorr
        self.corr_target = (1 - rho) * np.eye(ndim) \
            + rho * np.ones((ndim, ndim))

    def as_pandas(self):
        """Represent parameters as pandas objects.

        """
        univ = pd.DataFrame({'Persistence': self.persistence,
                             'Feedback': self.beta,
                             'Unconditional mean': self.volmean}).T
        corr = pd.Series({'Feedback': self.bcorr,
                          'Persistence': self.acorr + self.bcorr})
        return univ, corr

    def update_deco(self, theta=None):
        """Update DECO parameters.

        """
        self.acorr, self.bcorr = theta

    def __str__(self):
        """String representation.

        """
        univ, corr = self.as_pandas()
        show = '\n\nNumber of dimensions = %d' % self.ndim
        show += '\n\nParameters of univariate volatilities:\n'
        show += univ.to_string(float_format=lambda x: '%.2f' % x)
        show += '\n\nParameters of correlation model:\n'
        show += corr.to_string(float_format=lambda x: '%.2f' % x)
        show += '\nCorrelation target:\n' + np.array_str(self.corr_target)
        return show + '\n'

    def __repr__(self):
        """String representation.

        """
        return self.__str__()