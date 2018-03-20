# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 18:06:08 2018

@author: Adam

    stats()
        - function for analysing pd.DataFrames
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
