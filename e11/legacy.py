# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 18:00:09 2018

@author: Adam

"""
import numpy as np
import pandas as pd
import sspals as sspals
from .core import MEASUREMENT_ID

def chmx_sspals(high, low, dt=1e-9, **kwargs):
    """ Combine high and low gain data (chmx).  Re-analyse each to find t0 (cfd
        trigger) and the delayed fraction (DF = BC/ AC) for limits=[A, B, C].

        args:
            high            np.array() [2D]
            low             np.array() [2D]
            dt=1e-9         float64

        kwargs:
            n_bsub=100                         # number of points to use to find offset
            invert=True                        # assume a negative (PMT) signal
            validate=False                     # only return rows with a vertical range > min_range
            min_range=0.1                      # see above
            
            cfd_scale=0.8                      # cfd
            cfd_offset=1.4e-8
            cfd_threshold=0.04

            limits=[-1.0e-8, 3.5e-8, 6.0e-7]   # delayed fraction ABC
            corr=True                          # apply boundary corrections
 
            dropna=False                       # remove empty rows
            debug=False                        # nans in output? try debug=True.
        
        return:
            pd.DataFrame(['t0', 'AC', 'BC', 'DF']))
    """
    chmx = sspals.chmx(high, low, **kwargs)
    df = sspals.sspals(chmx, dt, **kwargs)
    df.index.rename(MEASUREMENT_ID, inplace=True)
    return df