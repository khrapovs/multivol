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

        self.bcorr = bcorr
        self.acorr = acorr
        self.rho = rho
