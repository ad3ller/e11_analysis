# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:27:09 2017

@author: Adam
"""
import os
import glob
import re
import h5py
import numpy as np
import pandas as pd
import IPython
from warnings import warn
from tqdm import tqdm

def run_file(base, rid, ftype="_data.h5", check=True):
    """ build path to data file using run ID

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
    # reverse the label list (intuitve behaviour?)
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

class H5Data(object):
    """ A tool for working with oskar hdf5 data.
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
        # datafile
        if not os.path.isfile(self.fil):
            # file not found
            raise IOError(self.fil + " file not found.")
        # open datafile and extract info
        with h5py.File(self.fil, 'r') as dfil:
            self.info = utf8_attrs(dict(dfil.attrs))
            self.groups = list(dict(dfil.items()).keys())
            self.num_groups = len(self.groups)
            self.squids = np.sort(np.array(self.groups).astype(int))
        # build log file if missing
        if self.out_dire is not None and log_file is not None:
            self.log_file = os.path.join(self.out_dire, log_file)
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
        """ read squid attributes and save as log.pkl.
        """
        tqdm_kwargs = dict([(key.replace('tqdm_', ''), val) for key, val in kwargs.items() if 'tqdm_' in key])
        with h5py.File(self.fil, 'r') as dfil:
            # read attributes from each squid
            all_vars = []
            for gr in tqdm(self.groups, **tqdm_kwargs):
                # read info
                info = dict(dfil[gr].attrs)
                info = utf8_attrs(info)
                all_vars.append(pd.DataFrame([info], index=[int(gr)]))
            log_df = pd.concat(all_vars)
            log_df.index.name = 'squid'
            log_df.sort_index(inplace=True)
            #remove duplicate squid column
            log_df.drop('SQUID', axis=1, inplace=True)
            if 'DATETIME' in log_df:
                #log_df.DATETIME = log_df.DATETIME.str.decode('utf-8')
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
    def squid_attrs(self, squid):
        """ get squid group attributes

            args:
                squid

            return:
                h5[squid].attributes     dict()
        """
        squid_str = str(squid)
        if squid_str not in self.groups:
            raise LookupError("Group " + squid_str + " not found.")
        else:
            with h5py.File(self.fil, 'r') as dfil:
                data = dfil['.']
                return utf8_attrs(dict(data[squid_str].attrs))

    def datasets(self, squid=1):
        """ get group datasets

            args:
                squid=1

            return:
                tuple of the names of datasets in h5[squid]
        """
        squid_str = str(squid)
        if squid_str not in self.groups:
            raise LookupError("Group " + squid_str + " not found.")
        else:
            with h5py.File(self.fil, 'r') as dfil:
                data = dfil['.']
                return tuple(data[squid_str].keys())

    ## array data (e.g., traces and images)
    def array(self, squids, dataset, axis=0, cache=None, **kwargs):
        """ load HDF5 array h5[squids][dataset] and its attributes.

            args:
                squids      squid ID value(s)    (int)
                            can be a single value or a list of squid values
                dataset     name of dataset      (str)
                axis=0      concatenation axis   (int)

            kwargs:
                info=False             Return dataset attributes.
                ignore_missing=False   Don't complain if data is not found.
                update=False           Don't load cache.
                tqdm_kwargs

            return:
                array (np.array) [info (dict)]

            Nb. For stacking images from multiple squids use axis=2.
        """
        tqdm_kwargs = dict([(key.replace('tqdm_', ''), val) for key, val in kwargs.items() if 'tqdm_' in key])
        get_info = kwargs.get('info', False)
        ignore_missing = kwargs.get('ignore_missing', False)
        #update = kwargs.get('update', False)
        squids = np.array([squids]).flatten()
        arr = []
        info = None
        with h5py.File(self.fil, 'r') as dfil:
            for sq in tqdm(squids, **tqdm_kwargs):
                squid_str = str(sq)
                if squid_str in dfil and dataset in dfil[squid_str]:
                    arr.append(np.array(dfil[squid_str][dataset]))
                    if get_info:
                        if info is None:
                            info = [pd.DataFrame(utf8_attrs(dict(dfil[squid_str][dataset].attrs)),
                                                 index=[sq])]
                        elif get_info == 'all':
                            info.append(pd.DataFrame(utf8_attrs(dict(dfil[squid_str][dataset].attrs)),
                                                     index=[sq]))
                elif not ignore_missing:
                    raise Exception("Error: " + dataset + " not found for squid " \
                                    + squid_str + ".  Use ignore_missing=True if you don't care.")
        arr = np.concatenate(arr, axis=axis)
        if arr.dtype == 'int16':
            # int16 is a bit restrictive
            arr = arr.astype(int)
        if info is not None:
            info = pd.concat(info)
            info.index.name = 'squid'
        if get_info:
            return arr, info
        return arr

    ## DataFrame data (e.g., stats)
    def df(self, squids, dataset, columns=None, cache=None, **kwargs):
        """ load HDF5 DataFrame data[squids][dataset][columns] and its attributes.

            squid can be a single value or a list of squid values.

            If columns=None then return all columns in the dataset.

            args:
                squids         ID value(s)          (int / list/ array)
                dataset        name of dataset      (str)
                columns=None   names of columns     (list)
                cache=None     cache filename       (str)

            kwargs:
                label=None             This can be useful when merging datasets.
                                       If True, add dataset name as an index to the columns.
                                       Or, if label type is string, then use label.
                info=True              Return  dataset attributes for first dataset.
                                       if info='all' then get info from every dataset.
                ignore_missing=False   Don't complain if data is not found.
                columns_astype_str=False
                                       Convert column names to str.
                tqdm_kwargs
            return:
                df (pd.DataFrame) [info (dict)]
        """
        tqdm_kwargs = dict([(key.replace('tqdm_', ''), val) for key, val in kwargs.items() if 'tqdm_' in key])
        label = kwargs.get('label', None)
        get_info = kwargs.get('info', False)
        ignore_missing = kwargs.get('ignore_missing', False)
        columns_astype_str = kwargs.get('columns_astype_str', False)
        update = kwargs.get('update', False)
        # initialise
        squids = np.sort(np.array([squids]).flatten())
        arr = []
        info = None
        # load cached file
        if cache is not None and self.out_dire is not None:
            cache_file = (os.path.splitext(cache)[0]) + '.df.pkl'
            cache_file = os.path.join(self.out_dire, cache_file)
        if not update and cache is not None and os.path.isfile(cache_file):
            df, info = pd.read_pickle(cache_file)
            # check cache
            cache_squids = np.unique(df.index.get_level_values('squid'))
            if not np.array_equal(squids, cache_squids):
                warn('Cached file squids do not match selection. Use update=True to reload.')
            if columns is not None and not np.array_equal(np.sort(columns),
                                                          np.sort(df.columns)):
                warn('Cached file columns do not match selection. Use update=True to reload.')
        else:
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
                            tmp['repeat'] = tmp.index
                            tmp['squid'] = sq
                            arr.append(tmp)
                            if get_info:
                                if info is None:
                                    info = [pd.DataFrame(utf8_attrs(dict(dfil[squid_str][dataset].attrs)),
                                                         index=[sq])]
                                elif get_info == 'all':
                                    info.append(pd.DataFrame(utf8_attrs(dict(dfil[squid_str][dataset].attrs)),
                                                             index=[sq]))
                    elif not ignore_missing:
                        raise Exception("Error: " + dataset + " not found for squid " \
                                        + squid_str + ".  Use ignore_missing=True if you don't care.")
            num_df = len(arr)
            if num_df == 0:
                raise Exception('No datasets found')
            df = pd.concat(arr, ignore_index=True)
            df = df.set_index(['squid', 'repeat'])
            # convert column names to str
            if columns_astype_str:
                df.columns = np.array(df.columns.values).astype(str)
            # extend multiindex using label
            if label:
                if not isinstance(label, str):
                    label = dataset
                lbl_0 = np.full_like(df.columns, label)
                df.columns = pd.MultiIndex.from_arrays([lbl_0, df.columns])
        # return
        if info is not None:
            info = pd.concat(info)
            info.index.name = 'squid'            
        # output
        if cache is not None and self.out_dire is not None:
            obj = (df, info)
            pd.to_pickle(obj, cache_file)
        if get_info:
            return df, info
        return df

    ## pickle things into/ out of out_dire
    def ls(self, full=False, report=False):
        """ list the pkl files in the output directory
        """
        if self.out_dire is None:
            raise Exception('out_dire is None')
        fils = glob.glob(os.path.join(self.out_dire, '*.pkl'))
        if report:
            print('Found %d *.pkl files in %s'%(len(fils), self.out_dire))
        if full:
            return fils
        fnames = [os.path.split(f)[1] for f in fils]
        return fnames

    def save(self, obj, fname, out_dire=None):
        """  save an object as a pickle file.
        """
        fname = (os.path.splitext(fname)[0]) + '.pkl'
        if out_dire is None:
            # default - the output directory
            out_dire = self.out_dire
        elif os.path.isabs(out_dire):
            # absolute path
            if not os.path.isdir(out_dire):
                raise Exception(out_dire + ' does not exist.')
        else:
            # relative path
            out_dire = self.sub_dire(out_dire)
        out_file = os.path.join(out_dire, fname)
        pd.to_pickle(obj, out_file)

    def read(self, fname):
        """ read an object stored in a pickle file.
        """
        if os.path.isabs(fname):
            # absolute path
            fil = fname
        elif self.out_dire is None:
            raise Exception('out_dire is None')
        else:
            #relative path
            fil = os.path.join(self.out_dire, fname)
        # check
        if not os.path.isfile(fil):
            raise Exception(fname + ' does not exist or not a file.')
        obj = pd.read_pickle(fil)
        return obj

    ## misc. tools
    def sub_dire(self, sub_dire, fname=None):
        """ Build path to a sub-directory of h5.out_dire.  Create if does not exist."""
        if self.out_dire is None:
            raise Exception('Cannot build h5.sub_dire because h5.out_dire is None.')
        else:
            path = os.path.join(self.out_dire, sub_dire)
            if not os.path.exists(path):
                os.makedirs(path)
            if fname is not None:
                path = os.path.join(path, fname)
            return path

    def pprint(self):
        """ print author and description info """
        author = self.info['Author']
        desc = self.info['Description'].replace('\n', '\n\t\t ')
        output = ("file: \t\t %s \n" + \
                  "size: \t\t %.2f MB \n" + \
                  "groups: \t %d \n" + \
                  "author: \t %s \n" + \
                  "description: \t %s")%(self.fil, self.size*1e-6, self.num_groups, author, desc)
        print(output)
