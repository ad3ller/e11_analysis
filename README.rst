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
    arr




.. parsed-literal::

    array([[ 0.00052656,  0.00054219,  0.00054219, ..., -0.00243437,
            -0.00242656, -0.00244219],
           [ 0.00046406,  0.00047188,  0.00046406, ..., -0.00259844,
            -0.00259063, -0.00261406],
           [ 0.00046406,  0.00045625,  0.00044063, ..., -0.00248125,
            -0.00250469, -0.00249688],
           ..., 
           [-0.00014531, -0.00014531, -0.00016094, ..., -0.00309844,
            -0.00309063, -0.00308281],
           [ 0.00047188,  0.00046406,  0.0004875 , ..., -0.00249688,
            -0.0025125 , -0.0025125 ],
           [ 0.00058125,  0.00056563,  0.00055781, ..., -0.00258281,
            -0.00258281, -0.00256719]])



.. code:: ipython3

    # DataFrame data
    df = scan.df('analysis')
    df.head()




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>var</th>
          <th>w0</th>
          <th>w1</th>
          <th>a0</th>
          <th>a1</th>
          <th>a2</th>
          <th>f</th>
        </tr>
        <tr>
          <th>measurement</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>32.0500</td>
          <td>777.950119</td>
          <td>786.992730</td>
          <td>0.000499</td>
          <td>-0.005882</td>
          <td>-1.118270</td>
          <td>-0.006381</td>
        </tr>
        <tr>
          <th>1</th>
          <td>32.0505</td>
          <td>777.950119</td>
          <td>786.992729</td>
          <td>0.000458</td>
          <td>-0.006048</td>
          <td>-1.122989</td>
          <td>-0.006506</td>
        </tr>
        <tr>
          <th>2</th>
          <td>32.0510</td>
          <td>777.950120</td>
          <td>786.992729</td>
          <td>0.000443</td>
          <td>-0.005974</td>
          <td>-1.138911</td>
          <td>-0.006417</td>
        </tr>
        <tr>
          <th>3</th>
          <td>32.0515</td>
          <td>777.950119</td>
          <td>786.992729</td>
          <td>0.000860</td>
          <td>-0.005568</td>
          <td>-0.995466</td>
          <td>-0.006428</td>
        </tr>
        <tr>
          <th>4</th>
          <td>32.0520</td>
          <td>777.950119</td>
          <td>786.992729</td>
          <td>0.000522</td>
          <td>-0.005977</td>
          <td>-1.095825</td>
          <td>-0.006499</td>
        </tr>
      </tbody>
    </table>
    </div>

See example notebooks.
