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

    def __init__(self):
        """Initialize parameter class.

        """
        self.ndim = 3
        self.persistence = .99
        self.beta = .85
        self.alpha = self.persistence - self.beta
        self.volmean = .2

        self.bcorr = .8
        self.acorr = .15
        self.prho = .9
