NumPy
=====

.. questions::

   - 1
   - 2
   - 3

.. objectives::

   - 1
   - 2
   - 3



Intro
-----

Being one of the most fundemental part of python scientific computing ecosystem, NumPy offers comprehensive mathematical functions, random number generators, linear algebra routines, Fourier transforms, and more. Moreover, NumPy is based on well-optimized C code, which gives much better performace than Python. (XXXX add vectorization, for this reason)


One has to import numpy to use it since it is not part of the Python standard library.

.. code:: python

       import numpy as np
       # find out what are the new functions in numpy
       dir(np)
       # find out what version we have
       np.__version__


So, we already know about python lists, and that we can put all kinds of things in there.
But in scientific usage, lists are often not enough. They are slow and not very flexible.


NDArray
-------

The core of numpy is the numpy ndarray (n-dimensional array).
A 1-dimentional array is a vector  
A 2-dimentional array is a matrix 

Compared to a python list, the numpy array is simialr in terms of serving as a data container.
Some differences between the two are: 

- numpy array can have multi dimensions 
- numpy array can work fast only when all data elements are of the same type  
- numpy array can be fast when vectorized  
- numpy array is slower for certain operations, e.g. appending elements 

Numpy Data Type
***************

The most common used data types (dtype) for numerical data (integer and floating-point) are listed here, 

For integers:

+-------------+----------------------------------+
| data type   | data range                       |
+=============+==================================+
| int8        | -2**7 to  2**7 -1                |
+-------------+----------------------------------+
| int16       | -32768 to 32767                  |
+-------------+----------------------------------+
| int32       | -2147483648 to 2147483647        |
+-------------+----------------------------------+
| int64       |    fff                           |
+-------------+----------------------------------+

For unsigned intergers:

+-------------+----------------------------------+
| data type   | data range                       |
+=============+==================================+
| uint8       | ffff                             |
+-------------+----------------------------------+
| uint16      | ffff                             |
+-------------+----------------------------------+
| uint32      | ffff                             |
+-------------+----------------------------------+
| uint64      |    fff                           |
+-------------+----------------------------------+


Be careful, once the data value is beyond the lower or upper bound of a certain data type, 
the value will be wrapped around and there is no warning:

.. code:: python

	>>> np.array([255], np.uint8) + 1   # 2**8-1 is INT_MAX for uint8  
	array([0], dtype=uint8)



For floating-point numbers:

+-------------+----------------------------------+
| data type   | data range                       |
+=============+==================================+
| float16     | fff	                         |
+-------------+----------------------------------+
| float32     | fff     			 |
+-------------+----------------------------------+
| float64     |    fff                           |
+-------------+----------------------------------+


Note: float128 is Unix OS, but not on Windows OS.
Like intergers, the floating-point numbers suffer from overflow errors as well.

Array Creation
**************

One way to create a numpy array is to convert from a python list, but make sure that the list is homogeneous (same data type) 
otherwise you will downgrade the performace of numpy array. 
Since appending elements to an existing array is slow, it is a common practice to preallocate the necessary space with np.zeros or np.empty
when converting from a python list is not possible.

.. code:: python

       a = np.array([1,2,3]) 


.. code:: python

        >>> np.array([1, 2, 3]).dtype      
        dtype('int32')                   # int32 on Windows, int64 on Linux and MacOS






Array Operations and Manipulations
**********************************

All the familiar arithemtic operators are applied on an element-by-element basis.


.. challenge:: Arithmetic

   .. tabs:: 

      .. tab:: 1D

             .. code-block:: py

			import numpy as np
                        a = np.array([1, 3, 5])
                        b = np.array([4, 5, 6])

             .. code-block:: py

			a + b

             .. figure:: img/np_add_1d_new.svg 

             .. code-block:: py

			a/b

             .. figure:: img/np_div_1d_new.svg 


      .. tab:: 2D

             .. code-block:: python

			import numpy as np
		        a = np.array([[1, 2, 3],
	               	   [4, 5, 6]])
		        b = np.array([10, 10, 10],
	               	   [10, 10, 10]])

			a + b                       # array([[11, 12, 13],
                                			 #        [14, 15, 16]]) 

             .. figure:: img/np_add_2d.svg 


Array Indexing
**************

Basic indexing is similar to python lists.

.. challenge:: index


   .. tabs:: 

      .. tab:: 1D

             .. code-block:: py

			import numpy as np
                        data = np.array([1,2,3,4,5,6,7,8])

             .. figure:: img/np_ind_0.svg 

             .. code-block:: py

			     # integer indexing 

             .. figure:: img/np_ind_integer.svg 

             .. code-block:: py

			     # fancy indexing 

             .. figure:: img/np_ind_fancy.svg 

             .. code-block:: python

			     # boolean indexing 

             .. figure:: img/np_ind_boolean.svg 


      .. tab:: 2D

             .. code-block:: python

			     import numpy as np
			     data = np.array([[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]])

             .. figure:: img/np_ind2d_data.svg 

             .. code-block:: python

			     # integer indexing

             .. figure:: img/np_ind2d_integer.svg 

             .. code-block:: python

			     # fancy indexing 

             .. figure:: img/np_ind2d_fancy.svg 

             .. code-block:: python

			     # boolean indexing 


             .. figure:: img/np_ind2d_boolean.svg 



Array Aggregation
*****************

.. challenge:: aggregation

Apart from aggregate all values, one can also aggregate across the rows or columns by using the axis parameter:

   .. tabs:: 


      .. tab:: 2D

             .. code-block:: py

			     # max 

             .. figure:: img/np_max_2d.svg 


             .. code-block:: py

			     # sum 

             .. figure:: img/np_sum_2d.svg 

 
             .. code-block:: py

			     # axis 

             .. figure:: img/np_min_2d_ax0.svg 
             .. figure:: img/np_min_2d_ax1.svg 



Array Reshaping
***************

.. challenge:: reshape

Sometimes, you need to change the dimension of an array. One of the most common need is to trasnposing the matrix during the dot product.
Switching the dimensions of a numpy array is also quite common in more advanced cases.

             .. code-block:: py

			import numpy as np
                        data = np.array([1,2,3,4,6,7,8,9,10,11,12])



             .. figure:: img/np_reshape0.svg 

             .. code-block:: py

			    data.reshape(4,3)

             .. figure:: img/np_reshape43.svg 

             .. code-block:: py

			     data.reshape(3,4)
 
             .. figure:: img/np_reshape34.svg 




add example, T of 1d array is not working

Use flatten as an alternative to ravel. What is the difference? (Hint: check which one returns a view and which a copy)



.. keypoints::

   - NumPy is a powerful library every scientist using python should know about, since many other libraries also use it internally.
   - Be aware of some NumPy specific peculiarities

