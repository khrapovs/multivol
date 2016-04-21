#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Testing suite for DCC.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np
import numpy.testing as npt

from dcc import DCC
from dcc.dcc_recursion import (dcc_recursion_python, dcc_recursion_numba,
                               corr_dcc_python, corr_dcc_numba)


class DCCTestCase(ut.TestCase):

    """Test DCC."""

    def test_recursion(self):
        """Test recursion."""


        nobs = 10
        ndim = 30
        persistence = .99
        beta = .85
        volmean = .2

        acorr = .05
        bcorr = .9
        rho = .5
        param = np.array([acorr, bcorr, -.01])

        const = np.eye(ndim) * (1 - acorr - bcorr)

        data, rho_series = DCC.simulate(nobs=nobs, ndim=ndim, volmean=volmean,
                                         persistence=persistence, beta=beta,
                                         acorr=acorr, bcorr=bcorr, rho=rho)

        data = data.values
        neg_data = data.copy()
        neg_data[neg_data > 0] = 0
        qmat1 = np.zeros((nobs, ndim, ndim))
        qmat2 = np.zeros((nobs, ndim, ndim))
        corr_dcc1 = np.zeros((nobs, ndim, ndim))
        corr_dcc2 = np.zeros((nobs, ndim, ndim))

        dcc_recursion_python(qmat1, const, data, neg_data, param)
        dcc_recursion_numba(qmat2, const, data, neg_data, param)

        npt.assert_array_equal(qmat1, qmat2)

        corr_dcc_python(corr_dcc1, qmat1)
        corr_dcc_python(corr_dcc2, qmat2)

        npt.assert_array_equal(corr_dcc1, corr_dcc2)


if __name__ == '__main__':

    ut.main()
