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
            mode='default'
                           'count'    = count
                           'abbr'     = mean, err
                           'default'  = count, mean, std, err
                           'full'     = count, mean, std, err,
                                        max, min, range, median

        return:
            pd.DataFrame()
    """
    mode = kwargs.get('mode', 'default')
    #check Series or DataFrame
    if isinstance(df, pd.Series):
        df_columns = [df.name]
    elif isinstance(df, pd.DataFrame):
        df_columns = df.columns.values
    else:
        raise Exception('df must be a pandas.Series or pandas.DataFrame.')
    # prevent exeption being raised if list of length==1 is passed to groupby
    if isinstance(groupby, str):
        df_columns = [c for c in df_columns if c != groupby]
    elif len(groupby) == 1:
        groupby = groupby[0]
        df_columns = [c for c in df_columns if c != groupby]
    else:
        groupby = list(groupby)
        df_columns = [c for c in df_columns if c not in groupby]
    gr = df.groupby(groupby)
    # output
    if mode == 'count':
        red = [gr.count()]
        stat_columns = ['count']
    elif mode == 'abbr':
        red = [gr.mean(), gr.std() * gr.count()**-0.5]
        stat_columns = ['mean', 'err']
    elif mode == 'default':
        red = [gr.count(), gr.mean(), gr.std(), gr.std() * gr.count()**-0.5]
        stat_columns = ['count', 'mean', 'std', 'err']
    elif mode == 'full':
        red = [gr.count(), gr.mean(), gr.std(), gr.std() * gr.count()**-0.5,
               gr.max(), gr.min(), gr.max() - gr.min(), gr.median()]
        stat_columns = ['count', 'mean', 'std', 'err', 'max', 'min', 'range', 'median']
    else:
        raise Exception('kwarg mode=' + mode + ' is not valid.')
    # MultiIndex column names
    # remove groupby elements from output columns
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
