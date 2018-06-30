# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:32:50 2018

@author: Adam
"""
import os
import numpy as np 
from e11 import H5Data

DIRE = os.path.join(os.getcwd(), 'notebooks', 'example_data')
FILS = [os.path.join(DIRE, f) for f in ['array_data.h5', 'laser_data.h5']]
# attempt to load example files
h0 = H5Data(FILS[0])
h1 = H5Data(FILS[1])

def test_exists():
    """ check example files exist"""
    for fil in FILS:
        assert os.path.isfile(fil)
        
def test_squid():
    """ check squid values"""
    sq0 = np.arange(1, 7)
    assert np.array_equal(h0.squids, sq0)
    sq1 = np.arange(1, 65)
    assert np.array_equal(h1.squids, sq1)

def test_osc():
    """ load oscilloscope data """
    squids = 1
    img = h0.array(squids, 'OSC_0')
    assert (25, 2502) == np.shape(img)

def test_img():
    """ load image data """
    squids = [1, 2]
    img = h0.array(squids, 'IMG', axis=2)
    assert (128, 128, 19) == np.shape(img)
    
def test_df():
    """ load dataframe data """
    df = h0.df(h0.squids, 'AV_0', label=None, ignore_missing=False)
    assert np.array_equal(df.columns, ['AB', 'CD', 'EF'])
    assert 151 == df['AB'].count()