#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DCC model data
==============

"""
from __future__ import print_function, division

import numpy as np
import pandas as pd

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
