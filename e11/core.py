# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:27:09 2017

@author: Adam

    run_file()
        - function to build path to a run file
    
    cashew()  
        - caching wrapper
       
    H5Scan
        - class for accessing hdf5 files without groups
    
    H5Data
        - class for accessing hdf5 files with groups
        
    statistics()
        - function for analysing pd.DataFrames
"""
import os
import glob
import inspect
import re
from functools import wraps
from datetime import datetime
import h5py
import numpy as np
import pandas as pd
import IPython
from tqdm import tqdm
from .tools import get_tqdm_kwargs, utf8_attrs, add_index

def run_file(base, rid, ftype="_data.h5", check=True):
    """ Build path to data file using run ID

        base/YYYY/MM/DD/[rid]/[rid]_[ftype]

        args:
            base
            rid
            ftype='_data.h5'
            check=True      - exists?

        return:
            path to rid h5 file
    """
    year = rid[:4]
    month = rid[4:6]
    day = rid[6:8]
    dire = os.path.join(base, year, month, day, rid)
    fil = os.path.join(dire, rid + ftype)
    if check:
        if not os.path.isdir(dire):
            # run ID directory not found
            raise IOError(dire + " folder not found.")
        elif not os.path.isfile(fil):
            # run ID file not found
            raise IOError(fil + " file not found.")
    return fil

def cashew(method):
    """ Decorator to cache method result as a pickle file. Inherits from method:

        arg[0]:
            h5             instance of H5Data()

        kwargs:
            cache=None     If cache is not None, save result to h5.cache_dire/[cache].[method].pkl,
                           or read from the file if it already exists.
            update=False   If update then overwrite cached file.
            info=False     Information about result. Use to inspect the settings of the
                           cache file.
    """
    @wraps(method)
    def wrapper(h5, *args, **kwargs):
        """ function wrapper
        """
        cache = kwargs.get('cache', None)
        update = kwargs.get('update', False)
        get_info = kwargs.get('info', False)
        # TODO check info option
        if cache is None:
            cache = False
        # info
        sig = inspect.signature(method)
        arg_names = list(sig.parameters.keys())
        arg_names = arg_names[1:-1] # drop self and kwargs
        arg_values = [arg.__name__ if hasattr(arg, '__name__') else arg for arg in args]
        info = dict(zip(arg_names, arg_values))
        info['method'] = method.__name__
        # config cache
        if cache:
            if h5.cache_dire is None:
                raise Exception('cannot cache file without h5.cache_dire')
            elif isinstance(cache, bool):
                # TODO flatten args to default cache name
                fname = method.__name__ + '.pkl'
            elif isinstance(cache, str):
                fname = (os.path.splitext(cache)[0]) + '.' + method.__name__ + '.pkl'
            else:
                raise Exception('kwarg cache must be str or True')
            cache_file = os.path.join(h5.cache_dire, fname)
        # read cache ...
        if not update and cache and os.path.isfile(cache_file):
            result, info = pd.read_pickle(cache_file)
            info['cache'] = cache_file
        # ... or apply func ...
        else:
            result = method(h5, *args, **kwargs)
            info['datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # ... and update cache.
            if cache:
                obj = (result, info)
                pd.to_pickle(obj, cache_file)
            info['cache'] = False
        # result
        if get_info:
            return result, info
        return result
    return wrapper

class H5Scan(object):
    """ A simple tool for working with simple hdf5 data.
    """
    def __init__(self, fil, out_dire=None, update=False):
        # data file
        self.fil = fil
        self.dire = os.path.dirname(self.fil)
        self.size = os.path.getsize(self.fil)
        # output folder
        if out_dire is not None:
            if os.path.isabs(out_dire):
                # absolute path
                if os.path.isdir(out_dire):
                    self.out_dire = out_dire
                else:
                    raise Exception(out_dire + ' does not exist.')
            else:
                # relative path
                self.out_dire = os.path.join(self.dire, out_dire)
                if not os.path.exists(self.out_dire):
                    response = input(self.out_dire + ' does not exist.  Create? [Y/n]: ')
                    create = response.upper() in ['Y', 'YES']
                    if create:
                        os.makedirs(self.out_dire)
                    else:
                        self.out_dire = None
                    IPython.core.display.clear_output()
        else:
            self.out_dire = out_dire
        # cache directory
        if self.out_dire is None:
            self.cache_dire = None
        else:
            self.cache_dire = self.sub_dire('cache')
        # datafile
        if not os.path.isfile(self.fil):
            # file not found
            raise IOError(self.fil + " file not found.")
        # open datafile and extract info
        with h5py.File(self.fil, 'r') as dfil:
            self.attrs = utf8_attrs(dict(dfil.attrs))
            self.datasets = list(dict(dfil.items()).keys())
            self.num_datasets = len(self.datasets)

    ## squid info
    def dataset_attrs(self, dataset):
        """ Get group attributes.

            args:
                dataset                    str

            return:
                h5[dataset ].attributes    dict()
        """
        if dataset not in self.datasets:
            raise LookupError("squid = " + dataset + " not found.")
        else:
            # get group attributes
            with h5py.File(self.fil, 'r') as dfil:
                data = dfil['.']
                return utf8_attrs(dict(data[dataset].attrs))


    ## array data (e.g., traces and images)
    @cashew
    def array(self, dataset, **kwargs):
        """ Load HDF5 array h5[dataset].

            args:
                dataset     Name of dataset      (str)

            kwargs:
                cache=None     If cache is not None, save result to h5.cache_dire/[cache].array.pkl,
                               or read from the file if it already exists.
                update=False   If update then overwrite cached file.
                info=False     Information about result. Use to check that settings of the
                               cache match expectation.

            return:
                array (np.array)

        """
        # initialise
        with h5py.File(self.fil, 'r') as dfil:
            if dataset in dfil:
                arr = np.array(dfil[dataset])
            else:
                raise Exception("Error: " + dataset + " not found.")
        if 'int' in str(arr.dtype):
            # int8 and int16 is a bit restrictive
            arr = arr.astype(int)
        return arr

    ## DataFrame data (e.g., stats)
    @cashew
    def df(self, dataset, columns=None, **kwargs):
        """ load HDF5 DataFrame data[dataset][columns].

            If columns=None then return all columns in the dataset.

            args:
                dataset        name of dataset             (str)
                columns=None   names of columns            (list)

            kwargs:
                label=None             This can be useful when merging datasets.
                                       If True, add dataset name as an index to the columns.
                                       Or, if label.dtype is string, then use label.
                columns_astype_str=False
                                       Convert column names to str.
                cache=None     If cache is not None, save result to h5.cache_dire/[cache].df.pkl,
                               or read from the file if it already exists.
                update=False   If update then overwrite cached file.
                info=False     Information about result. Use to check that settings of the
                               cache match expectation.

            return:
                df (pd.DataFrame)
        """
        label = kwargs.get('label', None)
        columns_astype_str = kwargs.get('columns_astype_str', False)
        # open file
        with h5py.File(self.fil, 'r') as dfil:
            if dataset in dfil:
                if columns is None:
                    df = pd.DataFrame(np.array(dfil[dataset]))
                else:
                    df = pd.DataFrame(np.array(dfil[dataset]))[columns]
                num_rows = len(df.index.values)
                if num_rows > 0:
                    df['measurement'] = df.index + 1
            else:
                raise Exception("Error: " + dataset + " not found.")
        df = df.set_index(['measurement'])
        # convert column names to str
        if columns_astype_str:
            df.columns = np.array(df.columns.values).astype(str)
        # extend multiindex using label
        if label:
            if not isinstance(label, str):
                label = dataset
            lbl_0 = np.full_like(df.columns, label)
            df.columns = pd.MultiIndex.from_arrays([lbl_0, df.columns])
        # output
        return df

    ## misc. tools
    def sub_dire(self, dire, fname=None):
        """ Build path to a sub-directory of h5.out_dire.  Create if does not exist."""
        if self.out_dire is None:
            raise Exception('Cannot build h5.sub_dire because h5.out_dire is None.')
        else:
            path = os.path.join(self.out_dire, dire)
            if not os.path.exists(path):
                os.makedirs(path)
            if fname is not None:
                path = os.path.join(path, fname)
            return path

    def pprint(self):
        """ print author and description info """
        output = ("file: \t\t %s \n" + \
                  "size: \t\t %.2f MB \n" + \
                  "datasets: \t %s")%(self.fil, self.size*1e-6, self.datasets)
        print(output)

class H5Data(object):
    """ For working with oskar hdf5 data.
    """
    def __init__(self, fil, log_file='log.pkl', out_dire=None, update=False):
        # data file
        self.fil = fil
        self.dire = os.path.dirname(self.fil)
        self.size = os.path.getsize(self.fil)
        # output folder
        if out_dire is not None:
            if os.path.isabs(out_dire):
                # absolute path
                if os.path.isdir(out_dire):
                    self.out_dire = out_dire
                else:
                    raise Exception(out_dire + ' does not exist.')
            else:
                # relative path
                self.out_dire = os.path.join(self.dire, out_dire)
                if not os.path.exists(self.out_dire):
                    response = input(self.out_dire + ' does not exist.  Create? [Y/n]: ')
                    create = response.upper() in ['Y', 'YES']
                    if create:
                        os.makedirs(self.out_dire)
                    else:
                        self.out_dire = None
                    IPython.core.display.clear_output()
        else:
            self.out_dire = out_dire
        # cache directory
        if self.out_dire is None:
            self.cache_dire = None
        else:
            self.cache_dire = self.sub_dire('cache')
        # datafile
        if not os.path.isfile(self.fil):
            # file not found
            raise IOError(self.fil + " file not found.")
        # open datafile and extract info
        with h5py.File(self.fil, 'r') as dfil:
            self.attrs = utf8_attrs(dict(dfil.attrs))
            self.groups = list(dict(dfil.items()).keys())
            self.num_groups = len(self.groups)
            self.squids = np.sort(np.array(self.groups).astype(int))
        # build log file if missing
        if self.cache_dire is not None and log_file is not None:
            self.log_file = os.path.join(self.cache_dire, log_file)
            if not update and os.path.exists(self.log_file):
                self.log = pd.read_pickle(self.log_file)
            else:
                self.update(inplace=True)
        else:
            self.log_file = None
            self.update(inplace=True)
        # cache
        self._var = None
        self._rec = None

    ## log file
    def update(self, inplace=True, **kwargs):
        """ Read squid attributes and save as cache/log.pkl.
        """
        tqdm_kwargs = get_tqdm_kwargs(kwargs)
        with h5py.File(self.fil, 'r') as dfil:
            # read attributes from each squid
            all_vars = []
            for gr in tqdm(self.groups, **tqdm_kwargs):
                # read info
                attrs = utf8_attrs(dict(dfil[gr].attrs))
                all_vars.append(pd.DataFrame([attrs], index=[int(gr)]))
            log_df = pd.concat(all_vars)
            log_df.index.name = 'squid'
            log_df.sort_index(inplace=True)
            # remove duplicate squid column
            log_df.drop('SQUID', axis=1, inplace=True)
            if 'DATETIME' in log_df:
                log_df.DATETIME = pd.to_datetime(log_df.DATETIME)
                log_df['ELAPSED'] = (log_df.DATETIME - log_df.DATETIME.min())
        # save to pickle file
        if self.log_file is not None:
            log_df.to_pickle(self.log_file)
        # result
        if inplace:
            self.log = log_df
            # reset var and rec
            self._var = None
            self._rec = None
        else:
            return log_df

    @property
    def var(self):
        """ DataFrame of just the VAR values from the log file """
        if self._var is None:
            df = self.log.filter(regex="VAR:")
            num_cols = len(df.columns)
            if num_cols > 0:
                df.columns = [re.split('^VAR:', col)[1] for col in df.columns.values]
            self._var = df
        return self._var

    @property
    def rec(self):
        """ DataFrame of just the REC values from the log file """
        if self._rec is None:
            df = self.log.filter(regex="REC:")
            num_cols = len(df.columns)
            if num_cols > 0:
                df.columns = [re.split('^REC:', col)[1] for col in df.columns.values]
            self._rec = df
        return self._rec

    ## squid info
    def group_attrs(self, squid):
        """ Get group attributes.

            args:
                squid                    int/ str

            return:
                h5[squid].attributes     dict()
        """
        # check squid
        if isinstance(squid, str):
            squid_str = squid
        elif isinstance(squid, int):
            squid_str = str(squid)
        else:
            raise TypeError('squid.dtype must be int or str.')
        if squid_str not in self.groups:
            raise LookupError("squid = " + squid_str + " not found.")
        else:
            # get group attributes
            with h5py.File(self.fil, 'r') as dfil:
                data = dfil['.']
                return utf8_attrs(dict(data[squid_str].attrs))

    def datasets(self, squid=1):
        """ Get group datasets.

            args:
                squid=1

            return:
                tuple of the names of datasets in h5[squid]
        """
        squid_str = str(squid)
        if squid_str not in self.groups:
            raise LookupError("squid = " + squid_str + " not found.")
        else:
            with h5py.File(self.fil, 'r') as dfil:
                data = dfil['.']
                return tuple(data[squid_str].keys())

    def dataset_attrs(self, dataset, squid=1):
        """ Get dataset attributes.

            args:
                squid=1                            (int)

            return:
                h5[squid][dataset].attributes     dict()
        """
        squid_str = str(squid)
        if squid_str not in self.groups:
            raise LookupError("squid = " + squid_str + " not found.")
        else:
            with h5py.File(self.fil, 'r') as dfil:
                data = dfil['.']
                attributes = utf8_attrs(dict(data[squid_str][dataset].attrs))
                attributes['squid'] = squid
                return attributes

    ## array data (e.g., traces and images)
    @cashew
    def array(self, dataset, squids, axis=0, **kwargs):
        """ Load HDF5 array h5[squids][dataset] and its attributes.

            args:
                dataset     Name of dataset      (str)
                squids      Group(s)             (int / list/ array)
                axis=0      Concatenation axis   (int)

            kwargs:
                ignore_missing=False   Don't complain if data is not found.

                cache=None     If cache is not None, save result to h5.cache_dire/[cache].array.pkl,
                               or read from the file if it already exists.
                update=False   If update then overwrite cached file.
                info=False     Information about result. Use to check that settings of the
                               cache match expectation.

                tqdm_kwargs

            return:
                array (np.array) [info (dict)]

            Nb. For stacking images from multiple squids use axis=2.
        """
        tqdm_kwargs = get_tqdm_kwargs(kwargs)
        ignore_missing = kwargs.get('ignore_missing', False)
        # initialise
        if isinstance(squids, int):
            squids = [squids]
        arr = []
        with h5py.File(self.fil, 'r') as dfil:
            for sq in tqdm(squids, **tqdm_kwargs):
                squid_str = str(sq)
                if squid_str in dfil and dataset in dfil[squid_str]:
                    dat = np.array(dfil[squid_str][dataset])
                    arr.append(dat)
                elif not ignore_missing:
                    raise Exception("Error: " + dataset + " not found for squid " \
                                    + squid_str + ".  Use ignore_missing=True if you don't care.")
        arr = np.concatenate(arr, axis=axis)
        if 'int' in str(arr.dtype):
            # int8 and int16 is a bit restrictive
            arr = arr.astype(int)
        return arr

    ## DataFrame data (e.g., stats)
    @cashew
    def df(self, dataset, squids, columns=None, **kwargs):
        """ load HDF5 DataFrame data[squids][dataset][columns] and its attributes.

            squids can be a single value or a list of values.

            If columns=None then return all columns in the dataset.

            args:
                dataset        name of dataset             (str)
                squids         group(s)                    (int / list/ array)
                columns=None   names of columns            (list)

            kwargs:
                label=None             This can be useful when merging datasets.
                                       If True, add dataset name as an index to the columns.
                                       Or, if label.dtype is string, then use label.
                ignore_missing=False   Don't complain if data is not found.
                columns_astype_str=False
                                       Convert column names to str.

                cache=None     If cache is not None, save result to h5.cache_dire/[cache].df.pkl,
                               or read from the file if it already exists.
                update=False   If update then overwrite cached file.
                info=False     Information about result. Use to inspect the settings of the
                               cache file.

                tqdm_kwargs

            return:
                df (pd.DataFrame) [info (dict)]
        """
        tqdm_kwargs = get_tqdm_kwargs(kwargs)
        label = kwargs.get('label', None)
        ignore_missing = kwargs.get('ignore_missing', False)
        columns_astype_str = kwargs.get('columns_astype_str', False)
        # initialise
        if isinstance(squids, int):
            squids = [squids]
        arr = []
        # open file
        with h5py.File(self.fil, 'r') as dfil:
            # loop over squid values
            for sq in tqdm(squids, **tqdm_kwargs):
                squid_str = str(sq)
                if squid_str in dfil and dataset in dfil[squid_str]:
                    if columns is None:
                        tmp = pd.DataFrame(np.array(dfil[squid_str][dataset]))
                    else:
                        tmp = pd.DataFrame(np.array(dfil[squid_str][dataset]))[columns]
                    num_rows = len(tmp.index.values)
                    if num_rows > 0:
                        tmp['measurement'] = tmp.index + 1
                        tmp['squid'] = sq
                        arr.append(tmp)
                elif not ignore_missing:
                    raise Exception("Error: " + dataset + " not found for squid " \
                                    + squid_str + ".  Use ignore_missing=True if you don't care.")
        num_df = len(arr)
        if num_df == 0:
            raise Exception('No datasets found')
        df = pd.concat(arr, ignore_index=True)
        df = df.set_index(['squid', 'measurement'])
        # convert column names to str
        if columns_astype_str:
            df.columns = np.array(df.columns.values).astype(str)
        # extend multiindex using label
        if label:
            if not isinstance(label, str):
                label = dataset
            lbl_0 = np.full_like(df.columns, label)
            df.columns = pd.MultiIndex.from_arrays([lbl_0, df.columns])
        # output
        return df

    @cashew
    def apply(self, func, dataset, squids, **kwargs):
        """ Apply func to h5.[squids][dataset(s)].

            args:
                func           function to apply to data   (obj)
                dataset        name of dataset             (str)
                squids         group(s)                    (int / list/ array)

            kwargs:
                cache=None     If cache is not None, save result to h5.cache_dire/[cache].apply.pkl,
                               or read from the file if it already exists.
                update=False   If update then overwrite cached file.
                info=False     Information about result. Use to check that settings of the
                               cache match expectation.

                tqdm_kwargs

            result:
                func(datasets, **kwargs), [info]
        """
        tqdm_kwargs = get_tqdm_kwargs(kwargs)
        # initialise
        if isinstance(squids, int):
            squids = [squids]
        if not isinstance(dataset, list):
            dataset = [dataset]
        # initialise output
        result = []
        # open file
        with h5py.File(self.fil, 'r') as dfil:
            # loop over each squid
            for sq in tqdm(squids, unit='sq', **tqdm_kwargs):
                squid_str = str(sq)
                if all([ds in dfil[squid_str] for ds in dataset]):
                    data = [dfil[squid_str][ds] for ds in dataset]
                    df = func(data, **kwargs)
                    df['squid'] = sq
                    result.append(df)
        num_sq = len(result)
        if num_sq == 0:
            raise Exception('No data found for '+ dataset + '.')
        result = pd.concat(result)
        result = add_index(result, 'squid', prepend=True)
        # output
        return result

    ## pickle things into/ out of out_dire
    def ls(self, dire=None, regex='*', full=False, report=False):
        """ List the contents of out_dire/[dire=None].

            e.g., to list pickle files in the cache,
                h5.ls(dire='cache', regex='*.pkl')
        """
        # initial check
        if self.out_dire is None and dire is None:
            raise Exception('h5.out_dire and arg dire can`t both be None')
        # folder
        if dire is None:
            # default - the output directory
            folder = self.out_dire
        elif os.path.isabs(dire):
            # absolute path
            folder = dire
        else:
            # relative path
            folder = os.path.join(self.out_dire, dire)
        # check exists
        if not os.path.isdir(folder):
            raise Exception(folder + ' does not exist.')
        fils = glob.glob(os.path.join(folder, regex))
        if report:
            print('Found %d matches to %s in `%s`'%(len(fils), regex, folder))
        if full:
            return fils
        fnames = [os.path.split(f)[1] for f in fils]
        return fnames

    def to_pickle(self, obj, fname, dire=None, **kwargs):
        """  Save an object as a pickle file.
        """
        force = kwargs.get('force', False)
        fname = (os.path.splitext(fname)[0]) + '.pkl'
        # folder
        if dire is None:
            # default - the output directory
            folder = self.out_dire
        elif os.path.isabs(dire):
            # absolute path
            folder = dire
        else:
            # relative path
            folder = self.sub_dire(dire)
        # check
        if not os.path.isdir(folder):
            raise Exception(folder + ' does not exist.')
        # file
        out_file = os.path.join(folder, fname)
        if os.path.exists(out_file) and not force:
            response = input(out_file + ' already exists.  Overwrite? [Y/n]: ')
            overwrite = response.upper() in ['Y', 'YES']
            IPython.core.display.clear_output()
            if overwrite:
                pd.to_pickle(obj, out_file)
        else:
            pd.to_pickle(obj, out_file)

    def read_pickle(self, fname, dire=None):
        """ Read an object stored in a pickle file.
        """
        if os.path.isabs(fname):
            # absolute path
            fil = fname
        else:
            if dire is None:
                folder = self.out_dire
            elif os.path.isabs(dire):
                # dire is absolute path
                folder = dire
            elif self.out_dire is None:
                raise Exception('out_dire is None')
            else:
                # relative path
                folder = os.path.join(self.out_dire, dire)
            fil = os.path.join(folder, fname)
        # check
        if not os.path.isfile(fil):
            raise Exception(fname + ' does not exist or not a file.')
        obj = pd.read_pickle(fil)
        return obj

    ## misc. tools
    def sub_dire(self, dire, fname=None):
        """ Build path to a sub-directory of h5.out_dire.  Create if does not exist."""
        if self.out_dire is None:
            raise Exception('Cannot build h5.sub_dire because h5.out_dire is None.')
        else:
            path = os.path.join(self.out_dire, dire)
            if not os.path.exists(path):
                os.makedirs(path)
            if fname is not None:
                path = os.path.join(path, fname)
            return path

    def pprint(self):
        """ print author and description info """
        author = self.attrs['Author']
        desc = self.attrs['Description'].replace('\n', '\n\t\t ')
        output = ("file: \t\t %s \n" + \
                  "size: \t\t %.2f MB \n" + \
                  "groups: \t %d \n" + \
                  "author: \t %s \n" + \
                  "description: \t %s")%(self.fil, self.size*1e-6, self.num_groups, author, desc)
        print(output)

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
