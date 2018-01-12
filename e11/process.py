# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:49:52 2017

@author: Adam
"""
import os
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from .tools import utf8_attrs

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

def vrange(h5, dataset, squids=None, axis=1, window=None, cache=None, **kwargs):
    """ Calculate the vertical range of an array dataset, e.g., traces or images.

        args:
            squids=None    If squids is None, return data from ALL squids.
            axis=1         Apply along axis=axis.
            window=None    Tuple of (start, end) indexes of data to analyse.
            cache=None     If cache is not None, save result to h5.out_dire/[cache].vr.pkl,
                           or read from the file if it already exists.

        kwargs:
            info=False     Get settings and information.
            update=False   If update then overwrite cached file.
            iconvert=False Use yscale and yoffset attributes to convert int to dbl dtype.
                           -- for speed assume these are the same for every squid!
    """
    tqdm_kwargs = dict([(key.replace('tqdm_', ''), val) for key, val in kwargs.items() if 'tqdm_' in key])
    iconvert = kwargs.get('iconvert', False)
    get_info = kwargs.get('info', False)
    update = kwargs.get('update', False)
    # load cached file
    if cache is not None and h5.out_dire is not None:
        cache_file = (os.path.splitext(cache)[0]) + '.vr.pkl'
        cache_file = os.path.join(h5.out_dire, cache_file)
    if not update and cache is not None and os.path.isfile(cache_file):
        df, info = pd.read_pickle(cache_file)
    # compute vrange from raw data
    else:
        if squids is None:
            # use all squid values
            squids = h5.squids
        # record information about processing
        info = dict()
        info['squids'] = squids
        info['function'] = 'process.vrange()'
        info['dataset'] = dataset
        info['datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')    
        # setup output df
        data = []
        # open file
        with h5py.File(h5.fil, 'r') as dfil:
            # loop over each squid
            for sq in tqdm(squids, unit='sq', **tqdm_kwargs):
                squid_str = str(sq)
                if dataset in dfil[squid_str]:
                    arr = np.array(dfil[squid_str][dataset])
                    ndims = len(np.shape(arr))
                    if  ndims > 2:
                        # flatten to last dimensions, e.g., for images
                        arr = arr.reshape(-1, arr.shape[-1]).T
                    if arr.dtype == 'int8' or arr.dtype == 'int16':
                        # int8 and int16 are too restrictive
                        arr = arr.astype(int)
                    if window is not None:
                        # array subset
                        arr = arr[:, window[0]:window[1]]
                    rng = np.max(arr, axis=axis) - np.min(arr, axis=axis)
                    tmp = pd.DataFrame(rng, columns=['vrange'])
                    tmp['repeat'] = tmp.index
                    tmp['squid'] = sq
                    data.append(tmp)
        num_sq = len(data)
        if num_sq == 0:
            raise Exception('No data found for '+ dataset + '.')
        df = pd.concat(data, ignore_index=True)
        df = df.set_index(['squid', 'repeat'])
        # integer to double conversion
        if iconvert:
            # get scale
            with h5py.File(h5.fil, 'r') as dfil:
                squid_str = str(squids[0])
                cinf = pd.DataFrame(utf8_attrs(dict(dfil[squid_str][dataset].attrs)), index=[squids[0]])
            # rescale
            if 'yscale' not in cinf.keys():
                raise Exception('attribute `yscale` is required for iconvert.')
            else:
                # alles klar
                df['vrange'] = df['vrange'] * cinf['yscale'].values[0]
        # output
        if cache is not None:
            obj = (df, info)
            pd.to_pickle(obj, cache_file)
    if get_info:
        return df, info
    else:
        return df

def winsum(h5, dataset, window=(0,-1), squids=None, cache=None, **kwargs):
    """ Index an array dataset using a given window and sum along axis=1.

        args:
            squids=None    If squids is None, return data from ALL squids.
            window=(0, -1)
                           Tuple or a list of tuples of (start, end) indexes that define the windows.
            cache=None     If cache is not None, save result to h5.out_dire/[cache].ws.pkl,
                           or read from the file if it already exists.

        kwargs:
            info=False     Get settings and information.
            mean=False     Divide the window sum by the window size (i.e., calculate the mean).
            update=False   If update and fname exists then overwrite cached file.
            iconvert=False Use yscale and yoffset attributes to convert int to dbl dtype.
                           -- for speed assume these are the same for every squid!
    """
    tqdm_kwargs = dict([(key.replace('tqdm_', ''), val) for key, val in kwargs.items() if 'tqdm_' in key])
    iconvert = kwargs.get('iconvert', False)
    get_info = kwargs.get('info', False)
    update = kwargs.get('update', False)
    mean = kwargs.get('mean', False)
    if isinstance(window, tuple):
        window = [window]
    # load cached file
    if cache is not None and h5.out_dire is not None:
        cache_file = (os.path.splitext(cache)[0]) + '.ws.pkl'
        cache_file = os.path.join(h5.out_dire, cache_file)
    if not update and cache is not None and os.path.isfile(cache_file):
        df, info = pd.read_pickle(cache_file)
    # compute window sum from raw data
    else:
        if squids is None:
            # use all squid values
            squids = h5.squids
        # information about processing
        info = dict()
        info['function'] = 'process.winsum()'
        info['dataset'] = dataset
        info['squids'] = squids
        info['mean'] = mean
        info['datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        info['windows'] = window
        # setup output df
        data = []
        # open file
        with h5py.File(h5.fil, 'r') as dfil:
            # loop over each squid
            for sq in tqdm(squids, unit='sq', **tqdm_kwargs):
                squid_str = str(sq)
                if dataset in dfil[squid_str]:
                    arr = np.array(dfil[squid_str][dataset])
                    ndims = len(np.shape(arr))
                    if ndims != 2:
                        raise Exception('winsum() is designed for 2D arrays only.')
                    if arr.dtype == 'int8' or arr.dtype == 'int16':
                        # int8 and int16 are too restrictive
                        arr = arr.astype(int)
                    ws = []
                    nv = []
                    for win in window:
                        win_arr = arr[:, win[0]:win[1]]
                        num_reps, num_vals = np.shape(win_arr)
                        nv.append(num_vals)
                        tot = np.sum(win_arr, axis=1)
                        if mean:
                            tot = tot / num_vals
                        ws.append(tot)
                    ws = np.array(ws).T
                    tmp = pd.DataFrame(ws, columns=window)
                    tmp['repeat'] = tmp.index
                    tmp['squid'] = sq
                    data.append(tmp)
        num_sq = len(data)
        if num_sq == 0:
            raise Exception('No data found for '+ dataset + '.')
        df = pd.concat(data, ignore_index=True)
        df = df.set_index(['squid', 'repeat'])
        # integer to double conversion
        if iconvert:
            # get scale
            with h5py.File(h5.fil, 'r') as dfil:
                squid_str = str(squids[0])
                cinf = pd.DataFrame(utf8_attrs(dict(dfil[squid_str][dataset].attrs)), index=[squids[0]])
            # rescale
            if 'yscale' not in cinf.keys() or 'yoffset' not in cinf.keys():
                raise Exception('attributes `yscale` and `yoffset` are required for iconvert.')
            else:
                for num_vals, win in zip(nv, window):
                    # TODO - check this
                    if mean:
                        df[win] = df[win] * cinf['yscale'].values[0] + cinf['yoffset'].values[0]
                    else:
                        df[win] = df[win] * cinf['yscale'].values[0] + num_vals * cinf['yoffset'].values[0]
        # output
        if cache is not None and h5.out_dire is not None:
            obj = (df, info)
            pd.to_pickle(obj, cache_file)
    if get_info:
        return df, info
    else:
        return df
