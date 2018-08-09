e11_analysis
============

Python tools for analysis of experimental data stored in hdf5 files.

Install
-------

Install using setuptools,

.. code-block:: bash

   git clone https://github.com/ad3ller/e11_analysis
   cd ./e11_analysis
   python setup.py install

And run tests,

.. code-block:: bash

   pytest

The package can now be imported into python as *e11*.  


Usage
-----

.. code:: ipython3

    import os
    from e11 import H5Scan

.. code:: ipython3

    # data file
    fil = ".\\example_data\\microwave_scan.h5"
    os.path.exists(fil)

.. parsed-literal::

    True

.. code:: ipython3

    # list datasets
    scan = H5Scan(fil)
    scan.datasets

.. parsed-literal::

    ['analysis', 'osc_0']

.. code:: ipython3

    # array data
    arr = scan.array('osc_0')
    print(arr.shape)

.. parsed-literal::

    (140, 1000)
    
.. code:: ipython3

    # DataFrame data
    df = scan.df('analysis')
    print(df[['var', 'w0', 'f']].head())

.. parsed-literal::

                     var          w0         f
    measurement                               
    0            32.0500  777.950119 -0.006381
    1            32.0505  777.950119 -0.006506
    2            32.0510  777.950120 -0.006417
    3            32.0515  777.950119 -0.006428
    4            32.0520  777.950119 -0.006499

See example notebooks.
