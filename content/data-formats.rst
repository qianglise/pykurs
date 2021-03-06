Data formats with Pandas and Numpy
==================================

.. questions::

   - How do you store your data right now?
   - Are you doing data cleaning / preprocessing every time you load the data?

.. objectives::

   - Learn the distinguishing characteristics of different data formats.
   - Learn how Pandas to read and write data in a variety of formats.



Motivation and Scope
--------------------

HPC has become an indispensable tool for  climate science community, with the advance of mordern computing sytems (especially with accerlarators like GPUs), more and more data is produced even faster rate and legacy software tools for data analysis can not handle them efficiently. This even becomes a obstacle to scientific progress in some cases. This course focuses on high performace data analysis, a subset of computing in which the raw data from either climate model simulation or observation is to be transformed into understanding following the steps below:

    1. read the raw data
    2. perform some operations on the data, from very simple (e.g. take the mean) to very complex (e.g. train a deep neural network)
    3. visualize the results or write the output into a file

The bulk of the content is devoted to the very basics of earth science data analysis using the modern scientific Python ecosystem, including Numpy, Scipy, Pandas, Xarray and  performace enhancement using numba, dask, cuPy.


What is a data?
---------------

bit and byte
************

The smallest building block of storage in the computer is a **bit**, 
which stores either a 0 or 1.
Normally a number of 8 bits are combined in a group to make a **byte**. 
One byte (8 bits) can represent/hold at most 2**8 distint values.
Organising bytes in different ways could further represent different types of information, i.e. data.

Numerical Data
**************

Different numerial data types (integer and floating-point) can be encoded as bytes. The more bytes we use for each value, the more range or precision we get, however the more memory it takes. For example, integers stored with 1 byte (8 bits) have a range from [-128, 127], while with 2 bytes (16 bits), the ranges becomes  [-32768, 32767].
Integers are whole numbers and can be represented precisely given enough bytes. However, for floating-point numbers the decimal fractions simply can not be represented exactly as binary (base 2) fractions in most cases which is known as the representation error. Arithmetic operations will further propagate this error. That is why in scienctific computing, numerical algorithms have to be carefully chosen and floating-point numbers are usally allocated with 8 bytes (sometimes 4 bytes) to make sure the inaccuracy is under control and does not lead to unsteady solutions.

Note:
Many climate models or certain parts of the models have the option of using single precision, i.e. 4 bytes or 32 bits, for floating-point numbers in order to achieve better performance at a small cost to the accuracy.



Text Data
*********

When it comes to text data, the simplest character encoding 
is ASCII (American Standard Code for Information Interchange) and was the most 
common character encodings until 2008 when UTF-8 took over.
The orignal ASCII uses only 7 bits for representing each character/letter and therefore encodes only 128 specified characters. Later  it became common to use an 8-bit byte to store each character in memory, providing an extended ASCII. 
As computer becomes more powerful and  there is need for including more characters from other languages like Chinese, Greek, Arabic, etc. UTF-8  becomes the most common encoding nowadays and it uses minimum one byte up to four bytes per character. 


In real applications, the scientific data is more complex and usually contains both numerical and text data. 
There is no single file format that is good for every case to store the dataset.
Here we list a few of the data and file formats used in climate modelling community:

Tabular Data
************

A very common type of data is the so-called "tabular data". The data is structured typically into rows and columns. Each column usually have a name and a specific data type while each row is a distinct sample which provides data according to each column including missing value.
The simplest and most common way to save tablular data is via the so-called CSV (comma-separated values) file.



Grided Data
***********

Grided data is another very common type, and usually the numerical data is saved in a multi-dimentional rectangular grid.
Most probably it is saved in one of the following formats:

Hierarchical Data Format (HDF5) - Container for many arrays
Network Common Data Form (NetCDF) - Container for many arrays which conform to the NetCDF data model
Zarr - New cloud-optimized format for array storage

Metadata
********

Metadata consists of the information about the data. 
Different types of data may have different metadata conventions. 

In Earth and Environmental science, we are fortunate to have widespread robust practices around metdata. For NetCDF files, metadata can be embedded directly into the data files. The most common metadata convention is Climate and Forecast (CF) Conventions, commonly used with NetCDF data

    
.. csv-table::
   :widths: auto
   :delim: ;

   data format ; :doc:`pro`  ; :doc:`con` 
   CSV ; :doc:`map clause`; :doc:`map clause`
   Parquet ; :doc:`the effect of both a map-to and a map-from`; :doc:`the effect of both a map-to and a map-from`
   HDF5  ; :doc:`On entering the region, variables in the list`; :doc:`On entering the region, variables in the list`
   NetCDF4  ; :doc:`from variables in the list are copied into` ; :doc:`from variables in the list are copied into` 

.. +---------------------------+-----------------------------------------------+
   |                           |                                               |
   +===========================+===============================================+
   |  CSV                      | map clause                                    |
   +---------------------------+-----------------------------------------------+
   |  Parquet                  | the effect of both a map-to and a map-from    |
   +---------------------------+-----------------------------------------------+
   |  HDF5                     | On entering the region, variables in the list |
   |                           | are initialized on the device using the       |
   |                           | original values from the host                 |
   +---------------------------+-----------------------------------------------+
   |  NetCDF4                  | At the end of the target region, the values   |
   |                           | from variables in the list are copied into    |
   |                           | the original variables on the host. On        |
   |                           | entering the region, the initial value of the |
   |                           | variables on the device is not initialized    |
   +---------------------------+-----------------------------------------------+




CSV (comma-separated values)
****************************

.. admonition:: Key features

   - **Type:** Text format
   - **Packages needed:** numpy, pandas
   - **Space efficiency:** Bad
   - **Good for sharing/archival:** Yes
   - Tidy data:
       - Speed: Bad
       - Ease of use: Great
   - Array data:
       - Speed: Bad
       - Ease of use: Ok for one or two dimensional data. Bad for anything higher.
   - **Best use cases:** Sharing data. Small data. Data that needs to be human-readable. 

CSV is by far the most popular file format, as it is human-readable and easily shareable.
However, it is not the best format to use when you're working with big data.



.. important::

    When working with floating point numbers, you should be careful to save the data with enough decimal places so that you won't lose precision.

1. you may lose data precision simply because you do not save the data with enough decimals(check english)
2.
    
    For example, double-precision floating point numbers have `~16 decimal places of precision <https://en.wikipedia.org/wiki/Double-precision_floating-point_format>`__, but if you use normal Python to write these numbers, you can easily lose some of that precision.
    Let's consider the following example:
    
    .. code-block:: python

        import numpy as np
        test_number = np.sqrt(2)
        # Write the number in a file
        test_file = open('sqrt2.csv', 'w')
        test_file.write('%f' % test_number)
        test_file.close()
        # Read the number from a file
        test_file = open('sqrt2.csv', 'r')
        test_number2 = np.float64(test_file.readline())
        test_file.close()
        # Calculate the distance between these numbers
        print(np.abs(test_number - test_number2))

    CSV writing routines in Pandas and numpy try to avoid problems such as these by writing the floating point numbers with enough precision, but even they are not infallible.
    
    
  

    In our case some rows of ``dataset_csv`` loaded from CSV do not match the original ``dataset`` as the last decimal can sometimes be rounded due to `complex technical reasons <https://docs.python.org/3/tutorial/floatingpoint.html#representation-error>`__.

    Storage of these high-precision CSV files is usually very inefficient storage-wise.

    Binary files, where floating point numbers are represented in their native binary format, do not suffer from such problems.


Parquet
*******

.. important::

    Using Parquet requires `pyarrow-package <https://arrow.apache.org/docs/python>`__ to be installed.
    
    You can try installing PyArrow with
    
    .. code-block:: bash
    
        !pip install pyarrow
    
    or you can take this as a demo.

.. admonition:: Key features

   - **Type:** Binary format
   - **Packages needed:** pandas, pyarrow
   - **Space efficiency:** Great
   - **Good for sharing/archival:** Yes
   - Tidy data:
       - Speed: Good
       - Ease of use: Great
   - Array data:
       - Speed: Good
       - Ease of use: It's complicated
   - **Best use cases:** Working with big datasets in tidy data format. Archival of said data.

`Parquet <https://arrow.apache.org/docs/python/parquet.html>`__ is a standardized open-source columnar storage format that is commonly used for storing big data in machine learning.
Parquet is usable from many different languages (C, Java, Python, MATLAB, Julia, etc.).

We can work with Parquet files with `to_parquet- and read_parquet-functions <https://pandas.pydata.org/docs/user_guide/io.html#io-parquet>`__::

    dataset.to_parquet('dataset.parquet')
    dataset_parquet = pd.read_parquet('dataset.parquet')

Parquet can be used to store arbitrary data as well, but doing that is a bit more complicated so we won't do that here.


HDF5 (Hierarchical Data Format version 5)
*****************************************

.. admonition:: Key features

   - **Type:** Binary format
   - **Packages needed:** pandas, PyTables, h5py
   - **Space efficiency:** Good for numeric data.
   - **Good for sharing/archival:** Yes, if datasets are named well.
   - Tidy data:
       - Speed: Ok
       - Ease of use: Good
   - Array data:
       - Speed: Great
       - Ease of use: Good
   - **Best use cases:** Working with big datasets in array data format.

HDF5 is a high performance storage format for storing large amounts of data in multiple datasets in a single file.
It is especially popular in fields where you need to store big multidimensional arrays such as physical sciences.









NetCDF4 (Network Common Data Form version 4)
********************************************

.. important::

    
    A great NetCDF4 interface is provided by a `xarray-package <https://xarray.pydata.org/en/stable/getting-started-guide/quick-overview.html#read-write-netcdf-files>`__.
    
  
.. admonition:: Key features

   - **Type**: Binary format
   - **Packages needed:** pandas, netCDF4/h5netcdf, xarray
   - **Space efficiency:** Good for numeric data.
   - **Good for sharing/archival:** Yes.
   - Tidy data:
       - Speed: Ok
       - Ease of use: Good
   - Array data:
       - Speed: Good
       - Ease of use: Great
   - **Best use cases:** Working with big datasets in array data format. Especially useful if the dataset contains spatial or temporal dimensions. Archiving or sharing those datasets.

NetCDF4 is a data format that uses HDF5 as its file format, but it has standardized structure of datasets and metadata related to these datasets.
This makes it possible to be read from various different programs.

NetCDF4 is by far the most common format for storing large data from big simulations in physical sciences.

Working with array data is easy as well::

    # Write array data as NetCDF4
    xr.DataArray(data_array).to_netcdf('data_array.nc', engine='h5netcdf')
    # Read array data from NetCDF4
    data_array_xarray = xr.open_dataarray('data_array.nc', engine='h5netcdf')
    data_array_netcdf4 = data_array_xarray.to_numpy()
    data_array_xarray.close()

The advantage of NetCDF4 compared to HDF5 is that one can easily add other metadata e.g. spatial dimensions (``x``, ``y``, ``z``) or timestamps (``t``) that tell where the grid-points are situated.
As the format is standardized, many programs can use this metadata for visualization and further analysis.




Data has to be stored somewhere before you can analyse it:

1.harddisk
2.internet
3.cloud-based storage

The most popular file formats in climate modelling community are: 





What is a data format?
----------------------

Whenever you have data (e.g. measurement data, simulation results, analysis results), you'll need a way to store it.
This applies both when

1. you're storing the data in memory while you're working on it;
2. you're storing it to a disk for later work.

Let's consider this randomly generated dataset with various columns::

    import pandas as pd
    import numpy as np
    
    n_rows = 100000

    dataset = pd.DataFrame(
        data={
            'string': np.random.choice(('apple', 'banana', 'carrot'), size=n_rows),
            'timestamp': pd.date_range("20130101", periods=n_rows, freq="s"),
            'integer': np.random.choice(range(0,10), size=n_rows),
            'float': np.random.uniform(size=n_rows),
        },
    )

    dataset.info()

This DataFrame already has a data format: it is in the tidy data format!
In tidy data format we have multiple columns of data that are collected in a Pandas DataFrame.

..  image:: img/pandas/tidy_data.png

Let's consider another example::

    n = 1000

    data_array = np.random.uniform(size=(n,n))
    data_array


Here we have a different data format: we have a two-dimentional array of numbers!
This is different to Pandas DataFrame as data is stored as one contiguous block instead of individual columns.
This also means that the whole array must have one data type.


..  figure:: https://github.com/elegant-scipy/elegant-scipy/raw/master/figures/NumPy_ndarrays_v2.png

    Source: `Elegant Scipy <https://github.com/elegant-scipy/elegant-scipy>`__

Now the question is: can we store these datasets in a file in a way that **keeps our data format intact**?

For this we need a **file format** that supports our chosen **data format**.

Pandas has support for `many file formats <https://pandas.pydata.org/docs/user_guide/io.html>`__ for tidy data and Numpy has support for `some file formats <https://numpy.org/doc/stable/reference/routines.io.html>`__ for array data.
However, there are many other file formats that can be used through other libraries.

What to look for in a file format?
----------------------------------

When deciding which file format you should use for your program, you should remember the following:

**There is no file format that is good for every use case.**

Instead, there are various standard file formats for various use cases: 

.. figure:: https://imgs.xkcd.com/comics/standards.png

   Source: `xkcd #927 <https://xkcd.com/927/>`__.

Usually, you'll want to consider the following things when choosing a file format:

1. Is everybody else / leading authorities in my field using a certain format?
   Maybe they have good reasons for using it.
2. Is the file format good for my data format (is it fast/space efficient/easy to use)?
3. Do I need a human-readable format or is it enought to work on it using programming languages?
4. Do I want to archive / share the data or do I just want to store it while I'm working?


Using some of the most popular file formats
-------------------------------------------

CSV (comma-separated values)
****************************

.. admonition:: Key features

   - **Type:** Text format
   - **Packages needed:** numpy, pandas
   - **Space efficiency:** Bad
   - **Good for sharing/archival:** Yes
   - Tidy data:
       - Speed: Bad
       - Ease of use: Great
   - Array data:
       - Speed: Bad
       - Ease of use: Ok for one or two dimensional data. Bad for anything higher.
   - **Best use cases:** Sharing data. Small data. Data that needs to be human-readable. 

CSV is by far the most popular file format, as it is human-readable and easily shareable.
However, it is not the best format to use when you're working with big data.

Pandas has a very nice interface for writing and reading CSV files with `to_csv <https://pandas.pydata.org/docs/user_guide/io.html#io-store-in-csv>`__- and `read_csv <https://pandas.pydata.org/docs/user_guide/io.html#io-read-csv-table>`__-functions::

    dataset.to_csv('dataset.csv', index=False)

    dataset_csv = pd.read_csv('dataset.csv')

Numpy has `routines <https://numpy.org/doc/stable/reference/routines.io.html#text-files>`__ for saving and loading CSV files as arrays as well ::

    np.savetxt('data_array.csv', data_array)

    data_array_csv = np.loadtxt('data_array.csv')

.. important::

    When working with floating point numbers you should be careful to save the data with enough decimal places so that you won't lose precision.
    
    For example, double-precision floating point numbers have `~16 decimal places of precision <https://en.wikipedia.org/wiki/Double-precision_floating-point_format>`__, but if you use normal Python to write these numbers, you can easily lose some of that precision.
    Let's consider the following example:
    
    .. code-block:: python

        import numpy as np
        test_number = np.sqrt(2)
        # Write the number in a file
        test_file = open('sqrt2.csv', 'w')
        test_file.write('%f' % test_number)
        test_file.close()
        # Read the number from a file
        test_file = open('sqrt2.csv', 'r')
        test_number2 = np.float64(test_file.readline())
        test_file.close()
        # Calculate the distance between these numbers
        print(np.abs(test_number - test_number2))

    CSV writing routines in Pandas and numpy try to avoid problems such as these by writing the floating point numbers with enough precision, but even they are not infallible.
    We can check whether our written data matches the generated data:
    
    .. code-block:: python

        dataset.compare(dataset_csv)

        np.all(data_array == data_array_csv) 

    In our case some rows of ``dataset_csv`` loaded from CSV do not match the original ``dataset`` as the last decimal can sometimes be rounded due to `complex technical reasons <https://docs.python.org/3/tutorial/floatingpoint.html#representation-error>`__.

    Storage of these high-precision CSV files is usually very inefficient storage-wise.

    Binary files, where floating point numbers are represented in their native binary format, do not suffer from such problems.

Feather
*******

.. important::

    Using Feather requires `pyarrow-package <https://arrow.apache.org/docs/python>`__ to be installed.
    
    You can try installing pyarrow with
    
    .. code-block:: bash
    
        !pip install pyarrow
    
    or you can take this as a demo.

.. admonition:: Key features

   - **Type:** Binary format
   - **Packages needed:** pandas, pyarrow
   - **Space efficiency:** Good
   - **Good for sharing/archival:** No
   - Tidy data:
       - Speed: Great
       - Ease of use: Good
   - Array data:
       - Speed: -
       - Ease of use: -
   - **Best use cases:** Temporary storage of tidy data. 

`Feather <https://arrow.apache.org/docs/python/feather.html>`__ is a file format for storing data frames quickly.
There are libraries for Python, R and Julia.

We can work with Feather files with `to_feather- and read_feather-functions <https://pandas.pydata.org/docs/user_guide/io.html#io-feather>`__::

    dataset.to_feather('dataset.feather')
    dataset_feather = pd.read_feather('dataset.feather')

Feather is not a good format for storing array data, so we won't present an example of that here.


Parquet
*******

.. important::

    Using Parquet requires `pyarrow-package <https://arrow.apache.org/docs/python>`__ to be installed.
    
    You can try installing PyArrow with
    
    .. code-block:: bash
    
        !pip install pyarrow
    
    or you can take this as a demo.

.. admonition:: Key features

   - **Type:** Binary format
   - **Packages needed:** pandas, pyarrow
   - **Space efficiency:** Great
   - **Good for sharing/archival:** Yes
   - Tidy data:
       - Speed: Good
       - Ease of use: Great
   - Array data:
       - Speed: Good
       - Ease of use: It's complicated
   - **Best use cases:** Working with big datasets in tidy data format. Archival of said data.

`Parquet <https://arrow.apache.org/docs/python/parquet.html>`__ is a standardized open-source columnar storage format that is commonly used for storing big data in machine learning.
Parquet is usable from many different languages (C, Java, Python, MATLAB, Julia, etc.).

We can work with Parquet files with `to_parquet- and read_parquet-functions <https://pandas.pydata.org/docs/user_guide/io.html#io-parquet>`__::

    dataset.to_parquet('dataset.parquet')
    dataset_parquet = pd.read_parquet('dataset.parquet')

Parquet can be used to store arbitrary data as well, but doing that is a bit more complicated so we won't do that here.


HDF5 (Hierarchical Data Format version 5)
*****************************************

.. admonition:: Key features

   - **Type:** Binary format
   - **Packages needed:** pandas, PyTables, h5py
   - **Space efficiency:** Good for numeric data.
   - **Good for sharing/archival:** Yes, if datasets are named well.
   - Tidy data:
       - Speed: Ok
       - Ease of use: Good
   - Array data:
       - Speed: Great
       - Ease of use: Good
   - **Best use cases:** Working with big datasets in array data format.

HDF5 is a high performance storage format for storing large amounts of data in multiple datasets in a single file.
It is especially popular in fields where you need to store big multidimensional arrays such as physical sciences.

Pandas allows you to store tables as HDF5 with `PyTables <https://www.pytables.org/>`_, which uses HDF5 to write the files.
You can create a HDF5 file with `to_hdf- and `read_parquet-functions <https://pandas.pydata.org/docs/user_guide/io.html#io-hdf5>`__::

    dataset.to_hdf('dataset.h5', key='dataset', mode='w')
    dataset_hdf5 = pd.read_hdf('dataset.h5')

PyTables comes installed with the default Anaconda installation.

For writing data that is not a table, you can use the excellent `h5py-package <https://docs.h5py.org/en/stable/>`__::

    import h5py
    
    # Writing:

    # Open HDF5 file
    h5_file = h5py.File('data_array.h5', 'w')
    # Write dataset
    h5_file.create_dataset('data_array', data=data_array)
    # Close file and write data to disk. Important!
    h5_file.close()
    
    # Reading:
    
    # Open HDF5 file again
    h5_file = h5py.File('data_array.h5', 'r')
    # Read the full dataset
    data_array_h5 = h5_file['data_array'][()]
    # Close file
    h5_file.close()

h5py comes with Anaconda as well.


NetCDF4 (Network Common Data Form version 4)
********************************************

.. important::

    Using NetCDF4 requires `netCDF4 <https://unidata.github.io/netcdf4-python>`__- or `h5netcdf <https://github.com/h5netcdf/h5netcdf>`__-package to be installed.
    h5netcdf is often mentioned as being faster to the official netCDF4-package, so we'll be using it in the example.
    
    A great NetCDF4 interface is provided by a `xarray-package <https://xarray.pydata.org/en/stable/getting-started-guide/quick-overview.html#read-write-netcdf-files>`__.
    
    You can try installing these packages with
    
    .. code-block:: bash
    
        !pip install h5netcdf xarray
    
    or you can take this as a demo.

.. admonition:: Key features

   - **Type**: Binary format
   - **Packages needed:** pandas, netCDF4/h5netcdf, xarray
   - **Space efficiency:** Good for numeric data.
   - **Good for sharing/archival:** Yes.
   - Tidy data:
       - Speed: Ok
       - Ease of use: Good
   - Array data:
       - Speed: Good
       - Ease of use: Great
   - **Best use cases:** Working with big datasets in array data format. Especially useful if the dataset contains spatial or temporal dimensions. Archiving or sharing those datasets.

NetCDF4 is a data format that uses HDF5 as its file format, but it has standardized structure of datasets and metadata related to these datasets.
This makes it possible to be read from various different programs.

NetCDF4 is by far the most common format for storing large data from big simulations in physical sciences.

Using interface provided by ``xarray``::

    # Write tidy data as NetCDF4
    dataset.to_xarray().to_netcdf('dataset.nc', engine='h5netcdf')
    # Read tidy data from NetCDF4
    import xarray as xr
    dataset_xarray = xr.open_dataset('dataset.nc', engine='h5netcdf')
    dataset_netcdf4 = dataset_xarray.to_pandas()
    dataset_xarray.close()

Working with array data is easy as well::

    # Write array data as NetCDF4
    xr.DataArray(data_array).to_netcdf('data_array.nc', engine='h5netcdf')
    # Read array data from NetCDF4
    data_array_xarray = xr.open_dataarray('data_array.nc', engine='h5netcdf')
    data_array_netcdf4 = data_array_xarray.to_numpy()
    data_array_xarray.close()

The advantage of NetCDF4 compared to HDF5 is that one can easily add other metadata e.g. spatial dimensions (``x``, ``y``, ``z``) or timestamps (``t``) that tell where the grid-points are situated.
As the format is standardized, many programs can use this metadata for visualization and further analysis.

npy (numpy array format)
************************

.. admonition:: Key features

   - **Type**: Binary format
   - **Packages needed:** numpy
   - **Space efficiency:** Good.
   - **Good for sharing/archival:** No.
   - Tidy data:
       - Speed: -
       - Ease of use: -
   - Array data:
       - Speed: Great
       - Ease of use: Good
   - **Best use cases:** Saving numpy arrays temporarily.

If you want to temporarily store numpy arrays, you can use the `numpy.save <https://numpy.org/doc/stable/reference/generated/numpy.save.html>`__- and `numpy.load <https://numpy.org/doc/stable/reference/generated/numpy.load.html>`__-functions::

    np.save('data_array.npy', data_array)
    data_array_npy = np.load('data_array.npy')

There also exists `numpy.savez <https://numpy.org/doc/stable/reference/generated/numpy.savez.html>`__-function for storing multiple datasets in a single file::

    np.savez('data_arrays.npz', data_array0=data_array, data_array1=data_array)
    data_arrays = np.load('data_arrays.npz')
    data_arrays['data_array0']

For big arrays it's good idea to check other binary formats such as HDF5 or NetCDF4.

Exercise 1
----------

.. challenge::

    - Create the example dataframe ``dataset`` with:
    
      .. code-block:: python
      
          import pandas as pd
          import numpy as np

          n_rows = 100000

          dataset = pd.DataFrame(
              data={
                  'string': np.random.choice(('apple', 'banana', 'carrot'), size=n_rows),
                  'timestamp': pd.date_range("20130101", periods=n_rows, freq="s"),
                  'integer': np.random.choice(range(0,10), size=n_rows),
                  'float': np.random.uniform(size=n_rows),
              },
          )
    - Use the ``%timeit``-magic to calculate how long it takes to save / load the dataset as a CSV-file.

.. solution::

    .. code-block:: python
    
        %timeit dataset.to_csv('dataset.csv', index=False)
    
        %timeit dataset_csv = pd.read_csv('dataset.csv')

Exercise 2
----------

.. challenge::
      
    - Save the dataset ``dataset`` using a binary format of your choice.
    - Use the ``%timeit``-magic to calculate how long it takes to save / load the dataset.
    - Did you notice any difference in speed?

.. solution::

    .. code-block:: python
    

        %timeit dataset.to_hdf('dataset.h5', key='dataset', mode='w')

        %timeit dataset_hdf5 = pd.read_hdf('dataset.h5')

Exercise 3
----------

.. challenge::

    - Create a numpy array. Store it as a npy.
    - Read the dataframe back in and compare it to the original one. Does the data match?

.. solution::

   .. code-block:: python

      import numpy as np

      my_array = np.array(10)

      np.save('my_array.npy', my_array)
      my_array_npy = np.load('my_array.npy')
      np.all(my_array == my_array_npy)

Benefits of binary file formats
-------------------------------

Binary files come with various benefits compared to text files.

1. They can represent floating point numbers with full precision.
2. Storing data in binary format can potentially save lots of space.
   This is because you do not need to write numbers as characters.
   Additionally some file formats support compression of the data.
3. Data loading from binary files is usually much faster than loading from text files.
   This is because memory can be allocated for the data before data is loaded as the type of data in columns is known.
4. You can often store multiple datasets and metadata to the same file.
5. Many binary formats allow for partial loading of the data.
   This makes it possible to work with datasets that are larger than your computer's memory.

**Performance when writing tidy dataset:**

For the tidy ``dataset`` we had, we can test the performance of the different file formats:

+-------------+----------------+-----------------+----------------+
| File format | File size [MB] | Write time [ms] | Read time [ms] |
+=============+================+=================+================+
| CSV         | 4.571760       | 0.296015        | 0.072096       |
+-------------+----------------+-----------------+----------------+
| Feather     | 2.202471       | 0.013013        | 0.007742       |
+-------------+----------------+-----------------+----------------+
| Parquet     | 1.820971       | 0.009052        | 0.009052       |
+-------------+----------------+-----------------+----------------+
| HDF5        | 4.892181       | 0.037609        | 0.033721       |
+-------------+----------------+-----------------+----------------+
| NetCDF4     | 6.894043       | 0.073829        | 0.010776       |
+-------------+----------------+-----------------+----------------+

The relatively poor performance of HDF5-based formats in this case is due to the data being mostly one dimensional columns full of character strings.


**Performance when writing data array:**

For the array-shaped ``data_array`` we had, we can test the performance of the different file formats:

+-------------+----------------+-----------------+----------------+
| File format | File size [MB] | Write time [ms] | Read time [ms] |
+=============+================+=================+================+
| CSV         | 23.841858      | 0.647893        | 0.639863       |
+-------------+----------------+-----------------+----------------+
| npy         | 7.629517       | 0.009885        | 0.002539       |
+-------------+----------------+-----------------+----------------+
| HDF5        | 7.631348       | 0.012877        | 0.002737       |
+-------------+----------------+-----------------+----------------+
| NetCDF4     | 7.637207       | 0.018905        | 0.009876       |
+-------------+----------------+-----------------+----------------+

For this kind of a data, HDF5-based formats perform much better.


Things to remember
------------------

1. **There is no file format that is good for every use case.**
2. Usually, your research question determines which libraries you want to use to solve it.
   Similarly, the data format you have determines file format you want to use.
3. However, if you're using a previously existing framework or tools or you work in a specific field, you should prioritize using the formats that are used in said framework/tools/field.
4. When you're starting your project, it's a good idea to take your initial data, clean it, and store the results in a good binary format that works as a starting point for your future analysis.
   If you've written the cleaning procedure as a script, you can always reproduce it.
5. Throughout your work, you should use code to turn important data to human-readable format (e.g. plots, averages, ``DataFrame.head()``), not to keep your full data in a human-readable format.
6. Once you've finished, you should store the data in a format that can be easily shared to other people.


Other file formats
------------------

Pickle
******

.. admonition:: Key features

   - **Type**: Binary format
   - **Packages needed:** None (`pickle <https://docs.python.org/3/library/pickle.html>`__-module is included with Python).
   - **Space efficiency:** Ok.
   - **Good for sharing/archival:** No! See warning below.
   - Tidy data:
       - Speed: Ok
       - Ease of use: Ok
   - Array data:
       - Speed: Ok
       - Ease of use: Ok
   - **Best use cases:** Saving Python objects for debugging.

.. warning::

    Loading pickles that have been provided from untrusted sources is
    risky as they can contain arbitrary executable code.

`Pickle <https://docs.python.org/3/library/pickle.html>`__ is Python's own serialization library.
It allows you to store Python objects into a binary file, but it is not a format you will want to use for long term storage or data sharing.
It is best suited for debugging your code by saving the Python variables for later inspection::

    import pickle

    with open('data_array.pickle', 'wb') as f:
        pickle.dump(data_array, f)

    with open('data_array.pickle', 'rb') as f:
        data_array_pickle = pickle.load(f)


JSON (JavaScript Object Notation)
*********************************

.. admonition:: Key features

   - **Type**: Text format
   - **Packages needed:** None (`json <https://docs.python.org/3/library/json.html#module-json>`__-module is included with Python).
   - **Space efficiency:** Ok.
   - **Good for sharing/archival:** No! See warning below.
   - Tidy data:
       - Speed: Ok
       - Ease of use: Ok
   - Array data:
       - Speed: Ok
       - Ease of use: Ok
   - **Best use cases:** Saving Python objects for debugging.

JSON is another popular human-readable data format.
It is especially common when dealing with web applications (REST-APIs etc.).
However, when you're working with big data, you rarely want to keep your data in this format.

Similarly to other popular files, Pandas can write and read json files with `to_json- <https://pandas.pydata.org/docs/user_guide/io.html#io-json-writer>`_ and `read_json <https://pandas.pydata.org/docs/user_guide/io.html#io-json-reader>`_-functions::

    dataset.to_json('dataset.json')
    dataset_json = pd.read_csv('dataset.json')

However, JSON is often used to represent hierarchical data with multiple layers or multiple connections. 
For such data you might need to do a lot more processing.


Excel (binary)
**************

.. admonition:: Key features

   - **Type**: Text format
   - **Packages needed:** `openpyxl <https://openpyxl.readthedocs.io/en/stable/>`__ 
   - **Space efficiency:** Bad.
   - **Good for sharing/archival:** Maybe.
   - Tidy data:
       - Speed: Bad
       - Ease of use: Good
   - Array data:
       - Speed: Bad
       - Ease of use: Ok
   - **Best use cases:** Sharing data in many fields. Quick data analysis.

Excel is very popular in social sciences and economics.
However, it is `not a good format <https://www.bbc.com/news/technology-54423988>`__ for data science.

See Pandas' documentation on `working with Excel files <https://pandas.pydata.org/docs/user_guide/io.html#excel-files>`_.

Using Excel files with Pandas requires `openpyxl <https://openpyxl.readthedocs.io/en/stable/>`__-package to be installed.


See also
--------

- `Pandas' IO tools <https://pandas.pydata.org/docs/user_guide/io.html>`__ .
- `Tidy data comparison notebook <https://github.com/AaltoSciComp/python-for-scicomp/tree/master/extras/data-formats-comparison-tidy.ipynb>`__
- `Array data comparison notebook <https://github.com/AaltoSciComp/python-for-scicomp/tree/master/extras/data-formats-comparison-array.ipynb>`__


.. keypoints::

   - Pandas can read and write a variety of data formats.
   - There are many good, standard formats, and you don't need to create your own.
   - There are plenty of other libraries dedicated to various formats.
