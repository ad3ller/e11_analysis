# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:49:52 2017

@author: Adam

Functions for use with H5Data.apply()

"""
import numpy as np
import pandas as pd

# process array data

def vrange(data, **kwargs):
    """ Calculate the vertical range for an array dataset.

        args:
            data           h5.dataset

        kwargs:
            axis=1         Apply along axis=axis.
            window=None    Tuple of (start, end) indexes of data to analyse.
                           2D datasets only, e.g., repeat oscilloscope traces.

        return:
            vrange pd.DataFrame(index=repeat)
    """
    axis = kwargs.get('axis', 1)
    window = kwargs.get('window', None)
    if not isinstance(data, list):
        data = [data]
    num_datasets = len(data)
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
    df = pd.DataFrame(result.T, columns=['vrange_%d'%i for i in range(num_datasets)])
    # set index
    df['repeat'] = df.index + 1
    df = df.set_index(['repeat'])
    return df

def total(data, **kwargs):
    """ Calculate the total value for an array dataset.

        args:
            data           h5.dataset

        kwargs:
            axis=1         Apply along axis=axis.
            window=None    Tuple of (start, end) indexes of data to analyse.
                           2D datasets only, e.g., repeat oscilloscope traces.

        return:
            total pd.DataFrame(index=repeat)
    """
    axis = kwargs.get('axis', 1)
    window = kwargs.get('window', None)
    if not isinstance(data, list):
        data = [data]
    num_datasets = len(data)
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
    df = pd.DataFrame(result.T, columns=['total_%d'%i for i in range(num_datasets)])
    # set index
    df['repeat'] = df.index + 1
    df = df.set_index(['repeat'])
    return df

def mean(data, **kwargs):
    """ Calculate the mean value for an array dataset.

        args:
            data           h5.dataset

        kwargs:
            axis=1         Apply along axis=axis.
            window=None    Tuple of (start, end) indexes of data to analyse.
                           2D datasets only, e.g., repeat oscilloscope traces.

        return:
            mean pd.DataFrame(index=repeat)
    """
    axis = kwargs.get('axis', 1)
    window = kwargs.get('window', None)
    if not isinstance(data, list):
        data = [data]
    num_datasets = len(data)
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
    df = pd.DataFrame(result.T, columns=['mean_%d'%i for i in range(num_datasets)])
    # set index
    df['repeat'] = df.index + 1
    df = df.set_index(['repeat'])
    return df
