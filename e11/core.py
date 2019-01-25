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
import os
import warnings
import inspect
import re
from functools import wraps
from datetime import datetime
import IPython
import h5py
import numpy as np
import pandas as pd
from numbers import Number
from collections.abc import Iterable
from tqdm import tqdm
from .tools import sub_dire, utf8_attrs

# constants
MEASUREMENT_ID = "measurement"


def run_file(base, rid, ftype="_data.h5", check=True):
    """ Build path to data file using run ID.

        base/YYYY/MM/DD/[rid]/[rid][ftype]

        The first 8 characters of the run ID are assumed to be of
        the format YYYYMMDD.  The rest of the run ID could be a 
        complete timestamp, or the date appended by a padded integer,
        or a descriptive label, e.g.,

            YYYYMMDD_hhmmss, or YYYYMMDD_001, or YYYYMMDD_label

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
            raise OSError(f"{dire} not found")
        elif not os.path.isfile(fil):
            # run ID file not found
            raise OSError(f"{fil} not found")
    return fil


def cashew(method):
    """ Decorator to save or load method result to or from a pickle file.

        args:
            <passed to method>

        kwargs:
            cache=None              If cache is not None, save result to
                                    cache_file, or read from it if it exists.
            update_cache=False      Overwrite the cache.
            get_info=False          Get information about method / cache.

        notes:

        # file name
        if cache is an absolute path, matching *.pkl:
            cache_file = cache
        elif isinstance(cache, bool):
            cache_file = dire/[method.__name__].pkl
        elif isinstance(cache, str):
            cache_file = dire/cache.[method.__name__].pkl
        else:
            cache_file = None 

        # directory
        if hasattr(args[0], "cache_dire"):
            cache_dire = args[0].cache_dire
        else:
            cache_dire = os.getcwd()
    """
    @wraps(method)
    def wrapper(*args, **kwargs):
        """ function wrapper
        """
        cache = kwargs.pop("cache", None)
        update_cache = kwargs.pop("update_cache", False)
        get_info = kwargs.pop("get_info", False)
        # info
        sig = inspect.signature(method)
        arg_names = list(sig.parameters.keys())
        arg_values = []
        for a in args:
            if isinstance(a, (str, Number, Iterable)):
                arg_values.append(a)
            elif hasattr(a, "__name__"):
                arg_values.append(a.__name__)
            else:
                arg_values.append(a.__class__.__name__)
        info = dict(zip(arg_names, arg_values))
        info = {**info, **kwargs}
        info["method"] = method.__name__
        # config cache
        if cache:
            # absolute path
            if os.path.isabs(cache):
                cache_file = cache
                cache_dire, fname = os.path.split(cache_file)
            # relative path
            else:
                if hasattr(args[0], "cache_dire"):
                    cache_dire = args[0].cache_dire
                else:
                    cache_dire = os.getcwd()
                # file name
                if isinstance(cache, bool):
                    fname = method.__name__ + ".pkl"
                elif isinstance(cache, str):
                    fname = f"{os.path.splitext(cache)[0]}.{method.__name__}.pkl"
                else:
                    raise TypeError("kwarg cache dtype must be str or True")
                cache_file = os.path.join(cache_dire, fname)
            # checks
            if not os.path.exists(cache_dire):
                raise OSError(f"{cache_dire} not found")
            _, ext = os.path.splitext(cache_file)
            if ext != ".pkl":
                raise NameError(f"{fname} should have `.pkl` extension")
        # read cache ...
        if not update_cache and cache and os.path.isfile(cache_file):
            result, info = pd.read_pickle(cache_file)
            info["cache"] = cache_file
        # ... or apply func ...
        else:
            result = method(*args, **kwargs)
            info["datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # ... and update cache.
            if cache:
                obj = (result, info)
                pd.to_pickle(obj, cache_file)
                info["cache"] = cache_file
            else:
                info["cache"] = False
        # result
        if get_info:
            return result, info
        return result
    return wrapper


class H5Scan(object):
    """ For hdf5 files that contain only datasets (no groups), e.g.

        root/
        ├── scope_0
        ├── scope_1
        └── analysis
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
                    raise OSError(f"{out_dire} not found")
            else:
                # relative path
                self.out_dire = os.path.join(self.dire, out_dire)
                if not os.path.exists(self.out_dire):
                    if force:
                        # don't ask, use the force
                        os.makedirs(self.out_dire)
                    else:
                        response = input(f"{self.out_dire} does not exist. "
                                         "Create? [Y/N]: ")
                        create = response.upper() in ["Y", "YES"]
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
            self.cache_dire = sub_dire(self.out_dire, "cache")
        # datafile
        if not os.path.isfile(self.fil):
            # file not found
            raise OSError(f"{self.fil} not found")
        # open datafile and extract info
        with h5py.File(self.fil, "r") as dfil:
            self._attrs = utf8_attrs(dict(dfil.attrs))
            self._datasets = list(dict(dfil.items()).keys())
            self.num_datasets = len(self._datasets)

    def attrs(self, dataset=None):
        """ Get attributes.

            args:
                dataset=None              (str)

            return:
                dict()

                h5.attributes if dataset is None else h5[dataset].attributes
        """
        if dataset is None:
            return self._attrs
        elif dataset not in self._datasets:
            raise LookupError(f"{dataset} dataset not found")
        else:
            # get dataset attributes
            with h5py.File(self.fil, "r") as dfil:
                data = dfil["."]
                return utf8_attrs(dict(data[dataset].attrs))

    def datasets(self, update=False):
        """ Get names of datasets.

            args:
                update=False

            return:
                list
        """
        if self._datasets is None or update:
            with h5py.File(self.fil, "r") as dfil:
                data = dfil["."]
                self._datasets = list(data.keys())
        return self._datasets

    @cashew
    def array(self, dataset, **kwargs):
        """ Load array data.

            args:
                dataset        Name of dataset      (str)

            kwargs:
                cache=None              If cache is not None, save result to
                                        cache_file, or read from it if it exists.
                update_cache=False      Overwrite the cache.
                get_info=False          Get information about method / cache.

            return:
                numpy.ndarray, [cache_info]

        """
        # initialise
        with h5py.File(self.fil, "r") as dfil:
            if dataset in dfil:
                arr = np.array(dfil[dataset])
            else:
                raise LookupError(f"{dataset} dataset not found")
        if "int" in str(arr.dtype):
            # int8 and int16 is a bit restrictive
            arr = arr.astype(int)
        return arr

    @cashew
    def df(self, dataset, columns=None, **kwargs):
        """ Load DataFrame data.

            args:
                dataset        name of dataset             (str)
                columns=None   names of columns            (list)

                If columns=None then return all columns in the dataset.

            kwargs:
                label=None             This can be useful for merging datasets.
                                       If True, add dataset name as an index to
                                       the columns. Or, if label.dtype is
                                       string, then use label.
                columns_astype_str=False
                                       Convert column names to str.

                cache=None              If cache is not None, save result to
                                        cache_file, or read from it if it exists.
                update_cache=False      Overwrite the cache.
                get_info=False          Get information about method / cache.

            return:
                pandas.DataFrame, [cache_info]
        """
        label = kwargs.get("label", None)
        columns_astype_str = kwargs.get("columns_astype_str", False)
        # open file
        with h5py.File(self.fil, "r") as dfil:
            if dataset in dfil:
                if columns is None:
                    df = pd.DataFrame(np.array(dfil[dataset]))
                else:
                    df = pd.DataFrame(np.array(dfil[dataset]))[columns]
            else:
                raise LookupError(f"{dataset} dataset not found")
        df.index.name = MEASUREMENT_ID
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

    def pprint(self):
        """ print author and description info """
        print(f"file:     \t {self.fil}",
              f"size:     \t {self.size * 1e-6:.2f} MB",
              f"datasets: \t {self._datasets}", sep="\n")


class H5Data(object):
    """ 
    For hdf5 files that contain datasets within a single level 
    of numbered groups, e.g.

        root/
        ├── 0/
        │   ├── image
        │   ├── scope_0
        │   ├── scope_1
        │   └── analysis
        ├── 1/
        │   ├── image
        │   ├── scope_0
        │         ⋮
        └──
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
                    raise OSError(f"{out_dire} not found")
            else:
                # relative path
                self.out_dire = os.path.join(self.dire, out_dire)
                if not os.path.exists(self.out_dire):
                    if force:
                        os.makedirs(self.out_dire)
                    else:
                        response = input(f"{self.out_dire} does not exist. "
                                         "Create? [Y/N]: ")
                        create = response.upper() in ["Y", "YES"]
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
            self.cache_dire = sub_dire(self.out_dire, "cache")
            self.log_file = os.path.join(self.cache_dire, "log.pkl")
        # datafile
        if not os.path.isfile(self.fil):
            # file not found
            raise OSError(f"{self.fil} not found")
        # initialise log cache
        self._log = None
        # open datafile and extract info
        with h5py.File(self.fil, "r") as dfil:
            self._attrs = utf8_attrs(dict(dfil.attrs))
            self.groups = list(dict(dfil.items()).keys())
            self.squids = np.sort(np.array(self.groups).astype(int))

    @property
    def author(self):
        """ author info """
        return self._attrs["Author"]

    @property
    def desc(self):
        """ data description """
        return self._attrs["Description"]

    def update_log(self, cache=True, **kwargs):
        """ Assemble group attributes into self.log (pandas.DataFrame).
            
            args:
                cache=True     cache result to [cache_dire]/log.pkl

            kwargs:
                tqdm_kw        dict()
        
        """
        tqdm_kw = kwargs.get("tqdm_kw", {})
        with h5py.File(self.fil, "r") as dfil:
            # refresh file info
            self._attrs = utf8_attrs(dict(dfil.attrs))
            self.groups = list(dict(dfil.items()).keys())
            self.squids = np.sort(np.array(self.groups).astype(int))
            # read attributes from each group
            result = dict()
            for group in tqdm(self.groups, **tqdm_kw):
                # read info
                result[int(group)] = utf8_attrs(dict(dfil[group].attrs))
            # assemble log
            result = (pd
                      .DataFrame
                      .from_dict(result, orient="index")
                      .sort_index(axis=0)
                      .sort_index(axis=1))
            result.index.name = "squid"
            if "SQUID" in result:
                # remove duplicate squid column
                result = result.drop("SQUID", axis=1)
            if "ACQUIRE" not in result and "START" in result and "END" in result:
                result["ACQUIRE"] = result["END"] - result["START"]
            if "DATETIME" in result:
                result.DATETIME = pd.to_datetime(result.DATETIME)
                result["ELAPSED"] = (result.DATETIME - result.DATETIME.min())
        # save to pickle file
        if cache and self.log_file is not None:
            result.to_pickle(self.log_file)
        # result
        self._log = result

    def load_log(self, check=True):
        """ Load [cache_dire]/log.pkl """
        if self.log_file is None:
            raise TypeError("log_file is None")
        elif not os.path.exists(self.log_file):
            raise OSError(f"{self.log_file} not found")
        else:
            # read cached file
            log_df = pd.read_pickle(self.log_file)
            if check and len(log_df.index) != len(self.groups):
                warnings.warn("len(log_df.index) != len(self.groups). "
                              "Try update_log().")
            self._log = log_df
        return log_df

    @property
    def log(self):
        """ Group attributes (pandas.DataFrame).

            if _log is not None:
                return _log                               # class cache
            elif os.path.exists(log_file):
                return log_file                           # file cache
            else:
                return update_log()                       # rebuilt log
        """
        if self._log is None:
            try:
                self.load_log()
            except (OSError, TypeError):
                self.update_log()
        return self._log

    @property
    def var(self):
        """ VAR:* columns of the log """
        tmp = self.log.filter(regex="VAR:")
        num_cols = len(tmp.columns)
        if num_cols > 0:
            tmp.columns = [re.split("^VAR:", c)[1] for c in tmp.columns.values]
        return tmp

    @property
    def rec(self):
        """ REC:* columns of the log """
        tmp = self.log.filter(regex="REC:")
        num_cols = len(tmp.columns)
        if num_cols > 0:
            tmp.columns = [re.split("^REC:", c)[1] for c in tmp.columns.values]
        return tmp

    def attrs(self, squid=None, dataset=None):
        """ Get root/group/dataset attributes.

            args:
                squid=None                  (int/ str)
                dataset=None                (str)

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
            return self._attrs
        # check squid
        elif isinstance(squid, str):
            group = squid
        elif isinstance(squid, (int, np.integer)):
            group = str(squid)
        else:
            raise TypeError("squid.dtype must be int or str.")
        # check squid
        if group not in self.groups:
            raise LookupError(f"squid={group} not found.")
        if dataset is None:
            # get group attributes
            with h5py.File(self.fil, "r") as dfil:
                data = dfil["."]
                return utf8_attrs(dict(data[group].attrs))
        # dataset info
        elif not isinstance(dataset, str):
            raise TypeError("dataset must be a str.")
        else:
            with h5py.File(self.fil, "r") as dfil:
                data = dfil["."]
                if dataset in data[group]:
                    attributes = utf8_attrs(dict(data[group][dataset].attrs))
                    attributes["squid"] = squid
                    return attributes
                else:
                    raise LookupError(f"{dataset} not found for squid={group}")

    def datasets(self, squid=1):
        """ Get the names of the datasets in group=squid

            args:
                squid=1

            return:
                list
        """
        group = str(squid)
        if group not in self.groups:
            raise LookupError(f"squid={group} not found")
        else:
            with h5py.File(self.fil, "r") as dfil:
                data = dfil["."]
                return list(data[group].keys())

    @cashew
    def array(self, squids, dataset, axis=0, **kwargs):
        """ Load array data.

            args:
                squids      group(s)             (int / list / array)
                dataset     name of dataset      (str)
                axis=0      concatenation axis   (int or None [returns list])

            kwargs:
                convert_int=False       Convert integer data types to python int
                ignore_missing=False    Don't complain if data is not found.

                cache=None              If cache is not None, save result to
                                        cache_file, or read from it if it exists.
                update_cache=False      Overwrite the cache.
                get_info=False          Get information about method / cache.

                tqdm_kw        dict()

            return:
                numpy.ndarray, [cache_info]

            notes:
                To stack images from multiple squids use axis=2.
        """
        convert_int = kwargs.get("convert_int", False)
        ignore_missing = kwargs.get("ignore_missing", False)
        tqdm_kw = kwargs.get("tqdm_kw", {})
        # initialise
        if isinstance(squids, int):
            squids = [squids]
        arr = []
        with h5py.File(self.fil, "r") as dfil:
            for sq in tqdm(squids, **tqdm_kw):
                group = str(sq)
                if (group in dfil) and (dataset in dfil[group]):
                    dat = np.array(dfil[group][dataset])
                    if convert_int and "int" in str(dat.dtype):
                        dat = dat.astype(int)
                    arr.append(dat)
                elif not ignore_missing:
                    warnings.warn(f"missing dataset(s) for squid={group}")
        if axis is not None:
            arr = np.concatenate(arr, axis=axis)
        return arr

    @cashew
    def df(self, squids, dataset, columns=None, **kwargs):
        """ Load DataFrame data.

            args:
                squids         group(s)                    (int / list/ array)
                dataset        name of dataset             (str)
                columns=None   names of columns            (list)

                If columns=None then return all columns in the dataset.

            kwargs:
                label=None              This can be useful for merging datasets.
                                        If True, add dataset name as an index to
                                        the columns. Or, if label.dtype is
                                        string, then use label.
                ignore_missing=False    Don't complain if data is not found.
                columns_astype_str=False
                                        Convert column names to str.

                cache=None              If cache is not None, save result to
                                        cache_file, or read from it if it exists.
                update_cache=False      Overwrite the cache.
                get_info=False          Get information about method / cache.

                tqdm_kw        dict()

            return:
                pandas.DataFrame, [cache_info]
        """
        label = kwargs.get("label", None)
        ignore_missing = kwargs.get("ignore_missing", False)
        columns_astype_str = kwargs.get("columns_astype_str", False)
        tqdm_kw = kwargs.get("tqdm_kw", {})
        # initialise
        if isinstance(squids, int):
            squids = [squids]
        result = dict()
        # open file
        with h5py.File(self.fil, "r") as dfil:
            # loop over squid values
            for sq in tqdm(squids, **tqdm_kw):
                group = str(sq)
                if (group in dfil) and (dataset in dfil[group]):
                    tmp = pd.DataFrame(np.array(dfil[group][dataset]))
                    tmp.index.name = MEASUREMENT_ID
                    if columns is not None:
                        tmp = tmp[columns]
                    result[sq] = tmp
                elif not ignore_missing:
                    warnings.warn(f"missing dataset(s) for squid: {group}.")
        num = len(result)
        if num == 0:
            raise LookupError("No matching datasets found")
        result = pd.concat(result, names=["squid"])
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
                dataset        name of dataset(s)          (str / list/ array)

            kwargs:
                ignore_missing=False    Don't complain if data is not found.

                tqdm_kw                 dict()

                cache=None              If cache is not None, save result to
                                        cache_file, or read from it if it exists.
                update_cache=False      Overwrite the cache.
                get_info=False          Get information about method / cache.

            return:
                func(datasets, **kwargs), [info]
        """
        ignore_missing = kwargs.get("ignore_missing", False)
        tqdm_kw = kwargs.get("tqdm_kw", {})
        # initialise
        if isinstance(squids, int):
            squids = [squids]
        if isinstance(dataset, str):
            dataset = [dataset]
        if "keys" in kwargs and isinstance(kwargs["keys"], bool) and kwargs["keys"]:
            kwargs["keys"] = dataset
        # initialise output
        result = dict()
        # open file
        with h5py.File(self.fil, "r") as dfil:
            # loop over each squid
            for sq in tqdm(squids, unit="sq", **tqdm_kw):
                group = str(sq)
                if all([ds in dfil[group] for ds in dataset]):
                    data = [dfil[group][ds] for ds in dataset]
                    result[sq] = func(*data, **kwargs)
                elif not ignore_missing:
                    warnings.warn(f"missing dataset(s) for squid: {group}.")
        num = len(result)
        if num == 0:
            raise LookupError(f"No data found for {dataset}.")
        result = pd.concat(result, names=["squid"])
        return result

    def pprint(self):
        """ print author and description info """
        desc = self.desc.replace("\n", "\n             \t ")
        print(f"file:        \t {self.fil}",
              f"size:        \t {self.size * 1e-6:.2f} MB",
              f"num groups:  \t {len(self.groups)}",
              f"author:      \t {self.author}",
              f"description: \t {desc}", sep="\n")
