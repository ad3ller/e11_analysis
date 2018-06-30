# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:07:04 2018

@author: Adam
"""
import os
from e11 import H5Data
from e11.stats import statistics
from e11.process import vrange

DIRE = os.path.join(os.getcwd(), 'notebooks', 'example_data')
FIL = os.path.join(DIRE, 'array_data.h5')
h5 = H5Data(FIL)

def test_vrange():
    """ test apply vrange function to data"""
    df = h5.apply(vrange, h5.squids, 'OSC_0', info=False)
    assert 1.0416186031361576 == float(df.median())
    
def test_statistics():
    """ test statistics """
    df = h5.apply(vrange, h5.squids, 'OSC_0', info=False)
    df2 = h5.var.join(df)
    st = statistics(df2, groupby=h5.var.columns, mode='full')
    assert 6.778302994221072 == st.loc[600].mean()