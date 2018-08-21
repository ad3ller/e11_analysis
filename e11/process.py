# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:49:52 2017

@author: Adam

Functions for use with H5Data.apply()

"""
from collections import Iterable
import numpy as np
import pandas as pd
from .core import MEASUREMENT_ID
from .trigs import threshold_trigger

def process_array(*data, func, subset, keys=None, convert_int=True, **kwargs):
    """ Process array dataset using func.

        args:
            data              [np.array()]
            func              function to apply
            subset            slice the array data
        
        kwargs:
            keys=None         list of names of each dataset for use with MultiIndex
            convert_int=True  convert any integer types to int

        return:
            pd.DataFrame()
    """
    result = []
    for arr in data:
        if convert_int and 'int' in str(arr.dtype):
            arr = arr.astype(int)
        if subset is not None:
            arr = arr[subset]
        result.append(func(arr))
    df = pd.concat(result, axis=1, keys=keys)
    df.index.rename(MEASUREMENT_ID, inplace=True)
    return df

def vrange(*data, axis=1, subset=None, name='vrange', keys=None, **kwargs):
    """ Calculate the vertical range for an array dataset.

        args:
            data           [np.array]

        kwargs:
            axis=1         apply func along axis=axis.
            subset=None    slice the array data
            name="vrange"  column name
            keys=None      list of names for each dataset

        return:
            pd.DataFrame()
    """
    func = lambda arr : pd.Series(np.max(arr, axis=axis) - np.min(arr, axis=axis), name=name)
    df = process_array(*data, func=func, subset=subset, keys=keys, **kwargs)
    return df

def total(*data, axis=1, subset=None, name='total', keys=None, **kwargs):
    """ Calculate the total value for an array dataset.

        args:
            data           [np.array]

        kwargs:
            axis=1         apply func along axis=axis.
            subset=None    slice the array data
            name="total"   column name
            keys=None      list of names for each dataset

        return:
            pd.DataFrame()
    """
    func = lambda arr : pd.Series(np.sum(arr, axis=axis), name=name)
    df = process_array(*data, func=func, subset=subset, keys=keys, **kwargs)
    return df

def mean(*data, axis=1, subset=None, name='mean', keys=None, **kwargs):
    """ Calculate the mean value for an array dataset.

        args:
            data           [np.array]

        kwargs:
            axis=1         apply along axis=axis.
            subset=None    slice the array data
            name="mean"    column name
            keys=None      list of names for each dataset

        return:
            pd.DataFrame()
    """
    func = lambda arr : pd.Series(np.mean(arr, axis=axis), name=name)
    df = process_array(*data, func=func, subset=subset, keys=keys, **kwargs)
    return df

def median(*data, axis=1, subset=None, name='mean', keys=None, **kwargs):
    """ Calculate the median value for an array dataset.

        args:
            data           [np.array]

        kwargs:
            axis=1         apply along axis=axis.
            subset=None    slice the array data
            name="mean"    column name
            keys=None      list of names for each dataset

        return:
            pd.DataFrame()
    """
    func = lambda arr : pd.Series(np.median(arr, axis=axis), name=name)
    df = process_array(*data, func=func, subset=subset, keys=keys, **kwargs)
    return df

def mean_std(*data, axis=1, subset=None, columns=['mean', 'std'], keys=None, **kwargs):
    """ Calculate the mean and standard deviation for an array dataset.

        args:
            data           [np.array]

        kwargs:
            axis=1         apply along axis=axis.
            subset=None    slice the array data
            columns=['mean', 'std']
                           column names
            keys=None      list of names for each dataset

        return:
            pd.DataFrame()

        notes:
            numpy.std() does not include the Bessel correction
    """
    func = lambda arr : pd.DataFrame(np.array([np.mean(arr, axis=axis), np.std(arr, axis=axis)]).T, columns=columns)
    df = process_array(*data, func=func, subset=subset, keys=keys, **kwargs)
    return df

def triggers(arr, dt, min_level, min_width, subset=None, **kwargs):
    """ Search for trigger events in array data using a threshold condition.

        Nb. trigger only accepts 1 dataset at a time.

        args:
            arr             np.array() [2D]
            dt              float64
            min_level       float64
            min_width       float64
        
        kwargs:
            subset=None     slice the array data
            bksub=50        number of points to use for background subtraction
            invert=True     invert data (events are negative)
            transpose=False search along rows (transpose=False) or columns (transpose=True)
            name="event"    index name
        
        return:
            pd.DataFrame(['time', 'width', 'amplitude'])
    """
    if subset is not None:
        arr = arr[subset]
    df = threshold_trigger(arr, dt, min_level, min_width, **kwargs)
    return df