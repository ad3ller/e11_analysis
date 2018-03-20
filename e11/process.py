# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:49:52 2017

@author: Adam

Functions for use with H5Data.apply()

"""
import numpy as np
import pandas as pd
from .core import MEASUREMENT_ID
from .tools import clabel

# process array data
def vrange(data, **kwargs):
    """ Calculate the vertical range for an array dataset.

        args:
            data           [np.array]

        kwargs:
            axis=1         Apply along axis=axis.
            window=None    Tuple of (start, end) indexes of data to analyse.
                           2D datasets only, e.g., repeat oscilloscope traces.
            label='vrange' 
                           Output column label(s). dtype must be str or iterable.

        return:
            vrange pd.DataFrame(index=repeat)
    """
    axis = kwargs.get('axis', 1)
    window = kwargs.get('window', None)
    label = kwargs.get('label', 'vrange')
    if not isinstance(data, list):
        data = [data]
    result = []
    for ds in data:
        arr = np.array(ds)
        ndims = len(np.shape(arr))
        if  ndims > 2:
            # flatten to last dimensions, e.g., for images
            arr = arr.reshape(-1, arr.shape[-1]).T
        if 'int' in str(arr.dtype):
            # int8 and int16 are too restrictive
            arr = arr.astype(int)
        if window is not None:
            if ndims != 2:
                raise Exception('kwarg `window` can only be used with 2D array datasets.')
            # array subset
            arr = arr[:, window[0]:window[1]]
        rng = np.max(arr, axis=axis) - np.min(arr, axis=axis)
        result.append(rng)
    result = np.array(result)
    df = pd.DataFrame(result.T)
    df = clabel(df, label)
    # set index
    df.index.rename(MEASUREMENT_ID, inplace=True)
    return df

def total(data, **kwargs):
    """ Calculate the total value for an array dataset.

        args:
            data           [np.array]

        kwargs:
            axis=1         Apply along axis=axis.
            window=None    Tuple of (start, end) indexes of data to analyse.
                           2D datasets only, e.g., repeat oscilloscope traces.
            label='total' 
                           Output column label(s). dtype must be str or iterable.

        return:
            total pd.DataFrame(index=repeat)
    """
    axis = kwargs.get('axis', 1)
    window = kwargs.get('window', None)
    label = kwargs.get('label', 'total')
    if not isinstance(data, list):
        data = [data]
    result = []
    for ds in data:
        arr = np.array(ds)
        ndims = len(np.shape(arr))
        if  ndims > 2:
            # flatten to last dimensions, e.g., for images
            arr = arr.reshape(-1, arr.shape[-1]).T
        if 'int' in str(arr.dtype):
            # int8 and int16 are too restrictive
            arr = arr.astype(int)
        if window is not None:
            if ndims != 2:
                raise Exception('kwarg `window` can only be used with 2D array datasets.')
            # array subset
            arr = arr[:, window[0]:window[1]]
        tot = np.sum(arr, axis=axis)
        result.append(tot)
    result = np.array(result)
    df = pd.DataFrame(result.T)
    df = clabel(df, label)
    # set index
    df.index.rename(MEASUREMENT_ID, inplace=True)
    return df

def mean(data, **kwargs):
    """ Calculate the mean value for an array dataset.

        args:
            data           [np.array]

        kwargs:
            axis=1         Apply along axis=axis.
            window=None    Tuple of (start, end) indexes of data to analyse.
                           2D datasets only, e.g., repeat oscilloscope traces.
            label='mean' 
                           Output column label(s). dtype must be str or iterable.

        return:
            mean pd.DataFrame(index=repeat)
    """
    axis = kwargs.get('axis', 1)
    window = kwargs.get('window', None)
    label = kwargs.get('label', 'mean')
    if not isinstance(data, list):
        data = [data]
    result = []
    for ds in data:
        arr = np.array(ds)
        ndims = len(np.shape(arr))
        if  ndims > 2:
            # flatten to last dimensions, e.g., for images
            arr = arr.reshape(-1, arr.shape[-1]).T
        if 'int' in str(arr.dtype):
            # int8 and int16 are too restrictive
            arr = arr.astype(int)
        if window is not None:
            if ndims != 2:
                raise Exception('kwarg `window` can only be used with 2D array datasets.')
            # array subset
            arr = arr[:, window[0]:window[1]]
        av = np.mean(arr, axis=axis)
        result.append(av)
    result = np.array(result)
    df = pd.DataFrame(result.T)
    df = clabel(df, label)
    # set index
    df.index.rename(MEASUREMENT_ID, inplace=True)
    return df
