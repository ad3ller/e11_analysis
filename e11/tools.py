# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 16:01:32 2018

@author: Adam
"""
import pandas as pd

def utf8_attrs(info):
    """ convert bytes to utf8

        args:
            info   dict()

        return
            info   dict() (decoded to utf8)
    """
    for key, val in info.items():
        if isinstance(val, bytes):
            info[key] = val.decode('utf8')
    return info

def add_column_index(df, label='', position='first'):
    """ Add a level to MultiIndex columns.

        This can be useful when joining DataFrames with / without multiindex columns.

        >>> st = statistics(a_df)      # MultiIndex DataFrame
        >>> add_column_index(h5.var, label='VAR').join(st)

        args:
            df          object to add index level to    pd.DataFrame()
            label=''    value(s) of the added level(s)  str() / list(str)
            position=0
                        position of level to add        'first'/ 'last' or int

        return:
            df.copy() with pd.MultiIndex()
    """
    df2 = df.copy()
    # multiple labels?
    if isinstance(label, str):
        # ..nope...
        label = [label]
    # reverse the label list (more intuitve behaviour?)
    label = label[::-1]
    # position is first?
    if position == 'first':
        position = 0
    # if df is Series then convert to DataFrame
    if isinstance(df2, pd.Series):
        df2 = pd.DataFrame(df2)
    for lbl in label:
        # add a level for each label
        df2_columns = df2.columns.tolist()
        new_columns = []
        for col in df2_columns:
            if not isinstance(col, tuple):
                col = (col,)
            col = list(col)
            if position == 'last':
                col.append(lbl)
            else:
                col.insert(position, lbl)
            new_columns.append(tuple(col))
        df2.columns = pd.MultiIndex.from_tuples(new_columns)
    return df2

def rescale(arr, yscale, yoffset):
    """ rescale = arr * yscale + yoffset
    """
    return arr * yscale + yoffset
    