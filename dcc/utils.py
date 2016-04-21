# -*- coding: utf-8 -*-
"""
Helper functions

"""
from __future__ import print_function, division

import time
import contextlib


def format_time(t):
    """Format time for nice printing.

    Parameters
    ----------
    t : float
        Time in seconds

    Returns
    -------
    format template

    """
    if t > 60 or t == 0:
        units = 'min'
        t /= 60
    elif t > 1:
        units = 's'
    elif t > 1e-3:
        units = 'ms'
        t *= 1e3
    elif t > 1e-6:
        units = 'us'
        t *= 1e6
    else:
        units = 'ns'
        t *= 1e9
    return '%.1f %s' % (t, units)


@contextlib.contextmanager
def take_time(desc):
    """Context manager for timing the code.

    Parameters
    ----------
    desc : str
        Description of the code

    Example
    -------
    >>> with take_time('Estimation'):
    >>>    estimate()

    """
    t0 = time.time()
    yield
    dt = time.time() - t0
    print('%s took %s' % (desc, format_time(dt)))
