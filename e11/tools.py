# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 16:01:32 2018

@author: Adam
"""
import os
import glob
from numbers import Number
import numpy as np
import pandas as pd


def sub_dire(base, dire, fname=None):
    """ Build path to a base/dire.  Create if does not exist."""
    if base is None:
        raise ValueError(f"base={base} is not valid")
    else:
        path = os.path.join(base, dire)
        if not os.path.exists(path):
            os.makedirs(path)
        if fname is not None:
            path = os.path.join(path, fname)
        return path


def ls(dire, regex="*", full_output=True, report=False):
    """ List the contents of dire.

    e.g., to list pickle files in the cache,
        ls(h5.cache_dire, regex="*.pkl")
    """
    # folder
    if dire is None:
        raise Exception("Cannot read from None directory.")
    # check exists
    if not os.path.isdir(dire):
        raise Exception(f"{dire} does not exist.")
    fils = glob.glob(os.path.join(dire, regex))
    if report:
        print(f"Found {len(fils)} matches to {regex} in {dire}.")
    if full_output:
        return fils
    fnames = [os.path.split(f)[1] for f in fils]
    return fnames


def to_pickle(obj, dire, fname, overwrite=False, **kwargs):
    """ save `obj` as [dire]/[fname].pkl
    
    args:
        obj             python object to save
        dire            directory
        fname           file name
        overwrite=False  

    kwargs:
        [passed to pandas.to_pickle()]

    """
    fname, _ = os.path.splitext(fname)
    fname += ".pkl"
    fil = os.path.join(dire, fname)
    # checks
    if not os.path.isdir(dire):
        raise OSError(f"{dire} not found")
    elif os.path.exists(fil) and not overwrite:
        raise OSError(f"{fil} already exists.  Use overwrite=True")
    else:
        pd.to_pickle(obj, fil, **kwargs)


def read_pickle(dire, fname, **kwargs):
    """ read `obj` from [dire]/[fname].pkl
    
    args:
        obj             python object to save
        dire            directory
        fname           file name.

    kwargs:
        [passed to pandas.read_pickle()]

    """
    fname, _ = os.path.splitext(fname)
    fname += ".pkl"
    fil = os.path.join(dire, fname)
    # checks
    if not os.path.exists(fil):
        raise OSError(f"{fil} not found")
    else:
        pd.read_pickle(fil, **kwargs)


def t_index(time, dt=1.0, t0=0.0):
    """ Convert time to index using dt [and t0].
    """
    if isinstance(time, Number):
        return int(round((time - t0) / dt))
    elif isinstance(time, tuple):
        return tuple([int(round((t - t0) / dt)) for t in time])
    elif isinstance(time, list):
        return list([int(round((t - t0) / dt)) for t in time])
    elif isinstance(time, np.ndarray):
        return np.array([int(round((t - t0) / dt)) for t in time])
    else:
        raise TypeError("time must be a number or list of numbers.")


def utf8_attrs(info):
    """ Convert bytes to utf8

        args:
            info   dict()

        return:
            info   dict() (decoded to utf8)
    """
    for key, val in info.items():
        if isinstance(val, bytes):
            info[key] = val.decode("utf8")
    return info


def add_level(df, label, position="first"):
    """ Add a level to pd.MultiIndex columns.

    This can be useful when joining DataFrames with / without multiindex
    columns.

    >>> st = statistics(df, groupby="squid")      # MultiIndex DataFrame
    >>> add_level(h5.var, "VAR").join(st)

    args:
        df          object to add index level to    pd.DataFrame()
        label=      value(s) of the added level(s)  str() / list(str)
        position=0
                    position of level to add        "first", "last" or int

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
    if position == "first":
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
            if position == "last":
                col.append(lbl)
            else:
                col.insert(position, lbl)
            new_columns.append(tuple(col))
        df2.columns = pd.MultiIndex.from_tuples(new_columns)
    return df2


def df_from_dict_of_tuples(data, names=("value", "error")):
    """ Construct a DataFrame with MultiIndex columns 
    from a dict of tuples.
    
    args:
        data : dict
        Of the form {row_i : {col_i: (item_i, ...), ...}, ...}
        
        names=("value", "error") : tuple
        Names of the items in each entry.
        
    return:
        pandas.DataFrame
        
        +-------+-----------------+-----------------+
        |       |     col_1       |     col_2       |
        |       | name_1 | name_2 | name_1 | name_2 |
        |-------+-----------------+-----------------+
        | row_1 | item_1 | item_2 | item_1 | item_2 | 
        | row_2 | item_1 | item_2 | item_1 | item_2 |
        
    """
    tmp = pd.DataFrame.from_dict(data, orient="index")
    df = pd.DataFrame()
    num = len(names)
    for c in tmp.columns:
        df[list(zip([c] * num, names))] = tmp[c].apply(pd.Series)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df

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
    d0 = sign[np.nonzero(sign)[0][0]]  # first non-zero diff
    try:
        result = np.argwhere(sign == -d0).flatten()[n]
    except IndexError:
        result = -1
    return result
