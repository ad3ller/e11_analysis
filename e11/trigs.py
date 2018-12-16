# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 17:48:02 2018

@author: Adam

digital trigger methods

"""
import numpy as np
import pandas as pd
from .core import MEASUREMENT_ID

def threshold_trigger(arr, dt, min_level, min_width, bksub=50, invert=True, transpose=False, name="event"):
    """ Search for trigger events in array data using a threshold condition.

        args:
            arr             np.array() [2D]
            dt              float64
            min_level       float64
            min_width       float64
        
        kwargs:
            bksub=50        number of points to use for background subtraction
            invert=True     invert data (events are negative)
            transpose=False search along rows (transpose=False) or columns (transpose=True)
            name="event"    index name
        
        return:
            pd.DataFrame(["time", "width", "amplitude"])

    """
    assert len(arr.shape) == 2, "arr must be a 2D numpy array"
    if transpose:
        arr = arr.T
    min_x = min_width / dt
    # initialise array
    result = dict()
    for i, row in enumerate(arr):
        if bksub is not None:
            row = row - np.mean(row[:bksub])
        if invert:
            row = -row
        # where exceeds threshold
        threshold = row > min_level
        threshold[0] = threshold[-1] = False
        # where True goes to False (pulse start/ stop)
        rngs = np.where(np.diff(threshold))[0].reshape(-1, 2)
        # exclude pulses narrower than min_width
        pulse = rngs[rngs[:, 1] - rngs[:, 0] > min_x]
        if len(pulse) > 0:
            # output
            time = pulse[:, 0] * dt
            width = (pulse[:, 1] - pulse[:, 0]) * dt
            amp = [np.max(row[slice(*p)]) for p in pulse]
            df = pd.DataFrame(np.array([time, width, amp]).T, columns=(["time", "width", "amplitude"]))
            df.index.rename(name, inplace=True)
            result[i] = df
    return pd.concat(result, names=[MEASUREMENT_ID])