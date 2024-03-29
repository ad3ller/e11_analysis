e11 analysis
============

**Note.  This project is no longer being developed and has been archived.**

Python 3 tools for analyzing experimental data stored in hdf5 files.


Install
-------

Requirements:
    h5py, scipy, numpy, pandas, pyarrow, tqdm, pytest

Optional:
    xarray

Install all dependencies then install `e11_analysis` using setuptools,

.. code-block:: bash

   git clone https://github.com/ad3ller/e11_analysis
   cd ./e11_analysis
   python setup.py install

And run tests,

.. code-block:: bash

   pytest

The package can now be imported into python as `e11`.  


Quick Start
-----------

.. code:: ipython3

    >>> from e11 import H5Scan

.. code:: ipython3

    >>> # data file
    >>> fil = ".\\example_data\\microwave_scan.h5"
    >>> scan = H5Scan(fil)

.. code:: ipython3

    >>> # list datasets
    >>> print(scan.datasets())

.. parsed-literal::

    ['analysis', 'osc_0']

.. code:: ipython3

    >>> # array data
    >>> arr = scan.array('osc_0')
    >>> print(arr.shape)

.. parsed-literal::

    (140, 1000)
    
.. code:: ipython3

    >>> # DataFrame data
    >>> df = scan.df('analysis')
    >>> print(df[['var', 'w0', 'f']].head())

.. parsed-literal::

                     var          w0         f
    measurement                               
    0            32.0500  777.950119 -0.006381
    1            32.0505  777.950119 -0.006506
    2            32.0510  777.950120 -0.006417
    3            32.0515  777.950119 -0.006428
    4            32.0520  777.950119 -0.006499

See example notebooks.
