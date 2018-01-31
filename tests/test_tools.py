# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:07:20 2018

@author: Adam
"""

from e11.tools import get_tqdm_kwargs

def test_tqdm_kw():
    """ test kwarg filtering """
    kwargs = dict({'a':1, 'b':2, 'tqdm_a':3, 'c':6, 'tqdm_e':4})
    assert dict({'a':3, 'e':4}) == get_tqdm_kwargs(kwargs)