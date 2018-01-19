# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:49:52 2017

@author: Adam
"""
import numpy as np
import pandas as pd

def statistics(df, groupby='squid', **kwargs):
    """ Calculate the mean and standard error for a DataFrame grouped by groupby.

        The output is simular to

        >>> df.groupby('squid').describe()

        args:
            df                 pd.DataFrame()
            groupby='squid'    str/ list/ np.array()

        kwargs:
            mode='basic'
                           'count'  = count
                           'abbr'   = mean, err
                           'basic'  = + std, count
                           'full'   = + min, max, median and range

        return:
            pd.DataFrame()
    """
    mode = kwargs.get('mode', 'basic')
    #check Series or DataFrame
    if isinstance(df, pd.Series):
        df_columns = [df.name]
    elif isinstance(df, pd.DataFrame):
        df_columns = df.columns.values
    else:
        raise Exception('df must be a pandas.Series or pandas.DataFrame.')
    # remove groupby elements from output columns
    df_columns = [c for c in df_columns if c not in list(groupby)]
    # prevent exeption being raised if list of length==1 is passed to groupby
    if not isinstance(groupby, str):
        if len(groupby) == 1:
            groupby = groupby[0]
        else:
            groupby = list(groupby)
    gr = df.groupby(groupby)
    # output
    if mode == 'count':
        red = [gr.count()]
        stat_columns = ['count']
    elif mode == 'abbr':
        red = [gr.mean(), gr.std() * gr.count()**-0.5]
        stat_columns = ['mean', 'err']
    elif mode == 'basic':
        red = [gr.count(), gr.mean(), gr.std(), gr.std() * gr.count()**-0.5]
        stat_columns = ['count', 'mean', 'std', 'err']
    elif mode == 'full':
        red = [gr.count(), gr.mean(), gr.std(), gr.std() * gr.count()**-0.5,
               gr.max(), gr.min(), gr.max() - gr.min(), gr.median()]
        stat_columns = ['count', 'mean', 'std', 'err', 'max', 'min', 'range', 'median']
    else:
        raise Exception('kwarg mode=' + mode + ' is not valid.')
    # MultiIndex column names
    new_columns = []
    for sc in stat_columns:
        for cc in df_columns:
            if not isinstance(cc, tuple):
                cc = (cc,)
            tc = cc + (sc,)
            new_columns.append(tc)
    # combine measurements
    av = pd.concat(red, axis=1)
    av.columns = pd.MultiIndex.from_tuples(new_columns)
    # sort columns
    av = av[np.sort(av.columns.values)]
    return av

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
        if arr.dtype == 'int8' or arr.dtype == 'int16':
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
        if arr.dtype == 'int8' or arr.dtype == 'int16':
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
        if arr.dtype == 'int8' or arr.dtype == 'int16':
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
