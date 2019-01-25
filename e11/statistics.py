# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 18:06:08 2018

@author: Adam

    statistics()
        - function for analysing pd.DataFrames
"""
import numpy as np
import pandas as pd


def statistics(data, groupby=None, mode="default"):
    """ Calculate the mean and standard error for a DataFrame grouped by
        groupby.

        The output of,
        >>> statistics(data, groupby="squid")
        
        is simular to that for,
        >>> data.groupby("squid").describe()

        args:
            data               pandas.DataFrame() or pandas.Series()
            groupby=None       str/ list/ np.array()
            mode="default"
                           "count"    = count
                           "abbr"     = mean, err
                           "default"  = count, mean, std, err
                           "full"     = count, mean, std, err,
                                        max, min, range, median

        return:
            pd.DataFrame()
    """
    # check Series or DataFrame
    if isinstance(data, pd.Series):
        df_columns = [data.name]
    elif isinstance(data, pd.DataFrame):
        df_columns = data.columns.values
    else:
        raise TypeError("data must be a pandas.Series or pandas.DataFrame.")
    # no groups
    if groupby is None:
        gr = data
    # group data
    else:
        # prevent exception caused by list of length==1 being passed to groupby
        if isinstance(groupby, str):
            df_columns = [c for c in df_columns if c != groupby]
        elif len(groupby) == 1:
            groupby = groupby[0]
            df_columns = [c for c in df_columns if c != groupby]
        else:
            groupby = list(groupby)
            df_columns = [c for c in df_columns if c not in groupby]
        gr = data.groupby(groupby)
    # analysis
    if mode == "count":
        red = [gr.count()]
        stat_columns = ["count"]
    elif mode == "abbr":
        red = [gr.mean(), gr.std() * gr.count()**-0.5]
        stat_columns = ["mean", "err"]
    elif mode == "default":
        red = [gr.count(), gr.mean(), gr.std(), gr.std() * gr.count()**-0.5]
        stat_columns = ["count", "mean", "std", "err"]
    elif mode == "full":
        red = [gr.count(), gr.mean(), gr.std(), gr.std() * gr.count()**-0.5,
               gr.max(), gr.min(), gr.max() - gr.min(), gr.median()]
        stat_columns = ["count", "mean", "std", "err", "max", "min", "range",
                        "median"]
    else:
        raise ValueError(f"mode={mode} is not valid")
    # result
    # no groups
    if groupby is None:
        result = pd.concat(red, axis=1, keys=stat_columns).T
    # group data
    else:
        # MultiIndex column names
        new_columns = []
        for sc in stat_columns:
            for cc in df_columns:
                if not isinstance(cc, tuple):
                    cc = (cc,)
                tc = cc + (sc,)
                new_columns.append(tc)
        # combine measurements
        result = pd.concat(red, axis=1)
        result.columns = pd.MultiIndex.from_tuples(new_columns)
        # sort columns
        result = result[np.sort(result.columns.values)]
    return result
