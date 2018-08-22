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

"""
import sys
import os
import inspect
import re
from functools import wraps
from datetime import datetime
import h5py
import numpy as np
import pandas as pd
import IPython
from tqdm import tqdm
from .tools import sub_dire, get_tqdm_kwargs, utf8_attrs

# constants
MEASUREMENT_ID = 'measurement'

def run_file(base, rid, ftype="_data.h5", check=True):
    """ Build path to data file using run ID.

        base/YYYY/MM/DD/[rid]/[rid][ftype]

        Usually the run ID will be a complete timestamp or the date appended by a
        padded integer, e.g.,

            YYYYMMDD_hhmmss, or YYYYMMDD_001.

        args:
            base
            rid
            ftype='_data.h5'
            check=True          does file exist?

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
    def __init__(self, fil, out_dire=None, force=False):
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
                    if force:
                        # don't ask, use the force
                        os.makedirs(self.out_dire)
                    else:
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
            self.cache_dire = sub_dire(self.out_dire, 'cache')
        # datafile
        if not os.path.isfile(self.fil):
            # file not found
            raise IOError(self.fil + " file not found.")
        # open datafile and extract info
        with h5py.File(self.fil, 'r') as dfil:
            self.attributes = utf8_attrs(dict(dfil.attrs))
            self.datasets = list(dict(dfil.items()).keys())
            self.num_datasets = len(self.datasets)

    ## attributes
    def attrs(self, dataset=None):
        """ Get attributes.

            args:
                dataset=None              str

            return:
                dict()
                
                h5.attributes if dataset is None else h5[dataset].attributes
        """
        if dataset is None:
            return self.attributes
        elif dataset not in self.datasets:
            raise LookupError("dataset = " + dataset + " not found.")
        else:
            # get dataset attributes
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
            else:
                raise Exception("Error: " + dataset + " not found.")
        df.index.rename(MEASUREMENT_ID, inplace=True)
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
    def pprint(self):
        """ print author and description info """
        output = ("file: \t\t %s \n" + \
                  "size: \t\t %.2f MB \n" + \
                  "datasets: \t %s")%(self.fil, self.size*1e-6, self.datasets)
        print(output)

class H5Data(object):
    """ For working with hdf5 that contains groups.
    """
    def __init__(self, fil, out_dire=None, update_log=False, force=False):
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
                    if force:
                        os.makedirs(self.out_dire)
                    else:
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
            self.log_file = None
        else:
            self.cache_dire = sub_dire(self.out_dire, 'cache')
            self.log_file = os.path.join(self.cache_dire, 'log.pkl')
        # datafile
        if not os.path.isfile(self.fil):
            # file not found
            raise IOError(self.fil + " file not found.")
        # initialise class cache
        self._log = None
        self._var = None
        self._rec = None
        # open datafile and extract info
        with h5py.File(self.fil, 'r') as dfil:
            self.attributes = utf8_attrs(dict(dfil.attrs))
            self.groups = list(dict(dfil.items()).keys())
            self.num_groups = len(self.groups)
            self.squids = np.sort(np.array(self.groups).astype(int))
        # log file
        if update_log:
            self.update_log()

    @property
    def author(self):
        """ author info """
        return self.attributes['Author']

    @property
    def desc(self):
        """ data description """
        return self.attributes['Description']

    ## log file
    def update_log(self, inplace=True, cache=True, **kwargs):
        """ Read squid attributes and save as cache/log.pkl.
        """       
        tqdm_kwargs = get_tqdm_kwargs(kwargs)
        with h5py.File(self.fil, 'r') as dfil:
            # read attributes from each squid
            all_vars = []
            for group in tqdm(self.groups, **tqdm_kwargs):
                # read info
                attrs = utf8_attrs(dict(dfil[group].attrs))
                all_vars.append(pd.DataFrame([attrs], index=[int(group)]))
            log_df = pd.concat(all_vars)
            log_df.index.name = 'squid'
            log_df.sort_index(inplace=True)
            # remove duplicate squid column
            log_df.drop('SQUID', axis=1, inplace=True)
            if 'DATETIME' in log_df:
                log_df.DATETIME = pd.to_datetime(log_df.DATETIME)
                log_df['ELAPSED'] = (log_df.DATETIME - log_df.DATETIME.min())
        # save to pickle file
        if cache and self.log_file is not None:
            log_df.to_pickle(self.log_file)
        # result
        if inplace:
            self._log = log_df
            # reset var and rec
            self._var = None
            self._rec = None
        else:
            return log_df

    def load_log(self, inplace=True):
        """ Load cached log file.  Default file is [cache_dire]/log.pkl.

            To load a custom file.

            >>> h5.log_file = [path to file]
            >>> h5.load_log()
        """
        if self.log_file is None:
            raise Exception("log_file not defined")
        elif not os.path.exists(self.log_file):
            raise Exception("log_file not found")
        else:
            # read cached file
            log_df = pd.read_pickle(self.log_file)
            if inplace:
                self._log = log_df
            else:
                return log_df

    @property
    def log(self):
        """ DataFrame of the experiment log. 
            
            if _log is not None:
                return _log                               # class cache 
            elif os.path.exists(log_file):
                return log_file                           # file cache
            else:
                return supdate_log()                      # rebuilt log
        """
        if self._log is not None:
            return self._log
        else:
            try:
                self.load_log(inplace=True)
            except:
                self.update_log(inplace=True)
            finally:
                return self._log

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
    def attrs(self, squid=None, dataset=None):
        """ Get root/group/dataset attributes.

            args:
                squid=None                  int/ str
                dataset=None                str
            return:
                dict()
                
                if squid is None:
                    h5.attributes
                elif dataset is None:
                    h5[squid].attributes     
                else:
                    h5[squid][dataset].attributes
        """
        if squid is None:
            return self.attributes
        # check squid
        elif isinstance(squid, str):
            group = squid
        elif isinstance(squid, int) or isinstance(squid, np.integer):
            group = str(squid)
        else:
            raise TypeError('squid.dtype must be int or str.')
        # check squid
        if group not in self.groups:
            raise LookupError("squid = " + group + " not found.")
        if dataset is None:
            # get group attributes
            with h5py.File(self.fil, 'r') as dfil:
                data = dfil['.']
                return utf8_attrs(dict(data[group].attrs))
        # dataset info
        elif not isinstance(dataset, str):
            raise TypeError('dataset must be a str.')
        else:
            with h5py.File(self.fil, 'r') as dfil:
                data = dfil['.']
                if dataset in data[group]:
                    attributes = utf8_attrs(dict(data[group][dataset].attrs))
                    attributes['squid'] = squid
                    return attributes
                else:
                    raise LookupError("dataset = " + dataset + " not found for squid = ." + group)

    def datasets(self, squid=1):
        """ Get group datasets.

            args:
                squid=1

            return:
                list
                
                names of datasets in h5[squid]
        """
        group = str(squid)
        if group not in self.groups:
            raise LookupError("squid = " + group + " not found.")
        else:
            with h5py.File(self.fil, 'r') as dfil:
                data = dfil['.']
                return list(data[group].keys())

    ## array data (e.g., traces and images)
    @cashew
    def array(self, squids, dataset, axis=0, **kwargs):
        """ Load HDF5 array h5[squids][dataset] and its attributes.

            args:
                squids      Group(s)             (int / list/ array)
                dataset     Name of dataset      (str)
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
                group = str(sq)
                if (group in dfil) and (dataset in dfil[group]):
                    dat = np.array(dfil[group][dataset])
                    arr.append(dat)
                elif not ignore_missing:
                    raise Exception("Error: " + dataset + " not found for squid " \
                                    + group + ".  Use ignore_missing=True if you don't care.")
        arr = np.concatenate(arr, axis=axis)
        if 'int' in str(arr.dtype):
            # int8 and int16 is a bit restrictive
            arr = arr.astype(int)
        return arr

    ## DataFrame data (e.g., stats)
    @cashew
    def df(self, squids, dataset, columns=None, **kwargs):
        """ load HDF5 DataFrame data[squids][dataset][columns] and its attributes.

            squids can be a single value or a list of values.

            If columns=None then return all columns in the dataset.

            args:
                squids         group(s)                    (int / list/ array)
                dataset        name of dataset             (str)
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
        result = dict()
        # open file
        with h5py.File(self.fil, 'r') as dfil:
            # loop over squid values
            for sq in tqdm(squids, **tqdm_kwargs):
                group = str(sq)
                if (group in dfil) and (dataset in dfil[group]):
                    _df = pd.DataFrame(np.array(dfil[group][dataset]))
                    _df.index.name = MEASUREMENT_ID
                    if columns is not None:
                        _df = _df[columns]
                    result[sq] = _df
                elif not ignore_missing:
                    raise Exception("Error: " + dataset + " not found for squid " \
                                    + group + ".  Use ignore_missing=True if you don't care.")
        num = len(result)
        if num == 0:
            raise Exception('No datasets found')
        result = pd.concat(result, names=['squid'])
        # convert column names to str
        if columns_astype_str:
            result.columns = np.array(result.columns.values).astype(str)
        # extend multiindex using label
        if label:
            if isinstance(label, bool):
                label = dataset
            lbl_0 = np.full_like(result.columns, label)
            result.columns = pd.MultiIndex.from_arrays([lbl_0, result.columns])
        # output
        return result

    @cashew
    def apply(self, func, squids, dataset, **kwargs):
        """ Apply func to h5.[squids][dataset(s)].

            args:
                func           function to apply to data   (obj)
                squids         group(s)                    (int / list/ array)
                dataset        name of dataset             (str)

            kwargs:
                cache=None     If cache is not None, save result to h5.cache_dire/[cache].apply.pkl,
                               or read from the file if it already exists.
                update=False   If update then overwrite cached file.
                info=False     Information about result. Use to check that settings of the
                               cache match expectation.

                tqdm_kwargs

            return:
                func(datasets, **kwargs), [info]
        """
        tqdm_kwargs = get_tqdm_kwargs(kwargs)
        # initialise
        if isinstance(squids, int):
            squids = [squids]
        if not isinstance(dataset, list):
            dataset = [dataset]
        if 'keys' in kwargs and isinstance(kwargs['keys'], bool) and kwargs['keys']:
            kwargs['keys'] = dataset
        # initialise output
        result = dict()
        # open file
        with h5py.File(self.fil, 'r') as dfil:
            # loop over each squid
            for sq in tqdm(squids, unit='sq', **tqdm_kwargs):
                group = str(sq)
                if all([ds in dfil[group] for ds in dataset]):
                    data = [dfil[group][ds] for ds in dataset]
                    result[sq] = func(*data, **kwargs)
        num = len(result)
        if num == 0:
            raise Exception('No data found for ' + dataset + '.')
        result = pd.concat(result, names=['squid'])
        # output
        return result

    ## misc. tools
    def pprint(self):
        """ print author and description info """
        desc = self.desc.replace('\n', '\n\t\t ')
        output = ("file: \t\t %s \n" + \
                  "size: \t\t %.2f MB \n" + \
                  "num groups: \t %d \n" + \
                  "author: \t %s \n" + \
                  "description: \t %s")%(self.fil, self.size*1e-6, self.num_groups, self.author, desc)
        print(output)
