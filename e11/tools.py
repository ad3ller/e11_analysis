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

        return:
            info   dict() (decoded to utf8)
    """
    for key, val in info.items():
        if isinstance(val, bytes):
            info[key] = val.decode('utf8')
    return info

def add_index(df, name, values=None, **kwargs):
    """ Add an index to pd.DataFrame(), e.g., for combing run data
    
        >>> a = add_index(a, 'rid', "20180109_181053", prepend=True)
        >>> b = add_index(b, 'rid', "20180109_191151", prepend=True)
        >>> c = pd.concat([a, b])
    
        args:
            df             pd.DataFrame()
            name           column to make index
            values=None    Index value(s).
                           Needed if column does not exist in df.
        
        kwargs:
            prepend=False  If True, place new index first.
            
        return:
            df             pd.DataFrame()
        
    """
    prepend = kwargs.get('prepend', False)
    if values is not None:
        df[name] = values
    df = df.set_index(name, append=True)
    if prepend:
        # move name to the front of the index
        index_names = df.index.names.copy()
        index_names.insert(0, index_names.pop(-1))
        df = df.reorder_levels(index_names)
    return df

def add_column_index(df, label='', position='first'):
    """ Add a level to pd.MultiIndex columns.

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
    