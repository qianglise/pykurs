Intro
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
    
      CSV writing routines in Pandas and numpy try to avoid problems such as these by writing the floating point numbers with enough precision, but even they are not infallible.
    
    
  

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



The advantage of NetCDF4 compared to HDF5 is that one can easily add other metadata e.g. spatial dimensions (``x``, ``y``, ``z``) or timestamps (``t``) that tell where the grid-points are situated.
As the format is standardized, many programs can use this metadata for visualization and further analysis.



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
    
    CSV writing routines in Pandas and numpy try to avoid problems such as these by writing the floating point numbers with enough precision, but even they are not infallible.


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




See also
--------

- `Pandas' IO tools <https://pandas.pydata.org/docs/user_guide/io.html>`__ .
- `Tidy data comparison notebook <https://github.com/AaltoSciComp/python-for-scicomp/tree/master/extras/data-formats-comparison-tidy.ipynb>`__
- `Array data comparison notebook <https://github.com/AaltoSciComp/python-for-scicomp/tree/master/extras/data-formats-comparison-array.ipynb>`__


.. keypoints::

   - Pandas can read and write a variety of data formats.
   - There are many good, standard formats, and you don't need to create your own.
   - There are plenty of other libraries dedicated to various formats.
