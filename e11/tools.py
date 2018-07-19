# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 16:01:32 2018

@author: Adam
"""
import os
import glob
from numbers import Number
from collections import Iterable
import pandas as pd

def sub_dire(dire, name, file_name=None):
    """ Build path to a sub-directory of dire.  Create if does not exist."""
    if dire is None:
        raise Exception('Cannot build sub_dire because dire is None.')
    else:
        path = os.path.join(dire, name)
        if not os.path.exists(path):
            os.makedirs(path)
        if file_name is not None:
            path = os.path.join(path, file_name)
        return path

def ls(dire, regex='*', full_output=True, report=False):
    """ List the contents of dire.

        e.g., to list pickle files in the cache,
            ls(h5.cache_dire, regex='*.pkl')
    """
    # folder
    if dire is None:
        raise Exception('can`t read from None directory')
    # check exists
    if not os.path.isdir(dire):
        raise Exception(dire + ' does not exist.')
    fils = glob.glob(os.path.join(dire, regex))
    if report:
        print('Found %d matches to %s in `%s`'%(len(fils), regex, dire))
    if full_output:
        return fils
    fnames = [os.path.split(f)[1] for f in fils]
    return fnames

def t_index(time, dt, t0=0.0):
    """ convert time to index using dt [and t0].
    """
    if isinstance(time, Number):
        return int(round((time - t0) / dt))
    elif isinstance(time, Iterable):
        return tuple(int(round((t - t0) / dt)) for t in time)
    else:
        raise TypeError("time must be a number or list of numbers.")

def get_tqdm_kwargs(kwargs):
    """ filter kwargs to those prepended by `tqdm_` and strip.
    """
    return dict([(key.replace('tqdm_', ''), val) for key, val in kwargs.items() if 'tqdm_' in key])

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

def nth_dflip(arr, n=0):
    """ Index of the nth occurance of a flip in the gradient of arr.
    
        e.g., use to find loops in variables for linear scans 
    
        args:
            arr          np.array(dims=1)
            n=0          int
          
        return:
            int
    """
    sign = np.sign(np.diff(arr))
    idx = np.nonzero(sign)[0][0]
    d0 = sign[idx]
    try:
        return np.argwhere(sign == -d0)[0][n]
    except:
        return -1

def clabel(df, label):
    """ Relable columns of a DataFrame.

        args:
            df             pd.DataFrame
            labels=None    Column labels.  Defaults to func.__name__.
                           dtype must be str or iterable.
    """
    cols = df.columns
    if isinstance(label, str):
        if len(cols) > 1:
            new_cols = [label + "_" + str(c) for c in cols]
        else:
            new_cols = [label]
    elif isinstance(label, Iterable):
        if len(label) == len(df.columns):
            new_cols = label
        else:
            raise TypeError('length of label must match number of columns')
    else:
        raise TypeError('label must be str or iterable')
    df.columns = new_cols
    return df
