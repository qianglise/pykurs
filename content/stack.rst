Python scientific computing stack Stack
=======================================

.. objectives::

   - Understand the Numpy array object
   - Be able to use basic NumPy functionality
   - Understand enough of NumPy to seach for answers to the rest of your questions ;)


.. _intro:

intro
-----
https://numfocus.org/project/numpy
NumPy is a universal data structure that fundamentally enables data analysis in numerical computing by permitting the exchange of data between algorithms. NumPy is a foundational project for the Python scientific computing stack.

https://www.machinelearningplus.com/python/101-numpy-exercises-python/
8. What is missing in numpy?

So far we have covered a good number of techniques to do data manipulations with numpy. But there are a considerable number of things you can’t do with numpy directly. At least to my limited knowledge. Let me list a few:

    No direct function to merge two 2D arrays based on a common column.
    Create pivot tables directly
    No direct way of doing 2D cross tabulations.
    No direct method to compute statistics (like mean) grouped by unique values in an array.
    And more..



The irritatingly catchy tune has racked up so many views that Google has been forced to upgrade YouTube's infrastructure to cope. When YouTube was first developed, nobody ever imagined that a video would be watched more than 2 billion times, so the view count was stored using a signed 32-bit integer.

The maximum value of this number type, 2,147,483,647, is well known to C programmers as INT_MAX. Once INT_MAX is reached, attempting to record another view will normally roll over to -2,147,483,648.

YouTube isn't the only software that this number is a problem for. Unix systems record time values as the number of seconds since 00:00:00 UTC on January 1, 1970. 32-bit systems use a signed 32-bit integer for this, so they will wrap around 2,147,483,647 seconds after that date. Two billion seconds is about 68 years; on January 19, 2038, at 03:14:07 in the morning, 32-bit Unix clocks will roll over.

While not a problem for desktops and servers, which have already made the switch to using 64-bit counts for the time, it's possible that embedded systems will continue to be 32-bit even 24 years from now.

As for YouTube, upgrading to a 64-bit signed number means that Gangnam Style is safe until it reaches 9,223,372,036,854,775,807 views. That's not likely to occur for another 4 billion years or so.


https://jakevdp.github.io/blog/2014/05/09/why-python-is-slow/

https://pangeo.io/packages.html#best-practices-for-pangeo-projects

https://pangeo.io/packages.html#why-xarray-and-dask

https://github.com/elegant-scipy/elegant-scipy/blob/master/markdown/ch1.markdown
 it focuses on exploring the NumPy array, a data structure that underlies almost all numerical scientific computation in Python
We will use the pandas library, which builds on NumPy
pandas is a Python library for data manipulation and analysis, with particular emphasis on tabular and time series data.  Here, we will use it here to read in tabular data of mixed type
 Although NumPy has some support for mixed data types (called "structured arrays"), it is not primarily designed for this use case, which makes subsequent operations harder than they need to be.



https://www.tutorialspoint.com/numpy/numpy_advanced_indexing.htm
It is possible to make a selection from ndarray that is a non-tuple sequence, ndarray object of integer or Boolean data type, or a tuple with at least one item being a sequence object. Advanced indexing always returns a copy of the data. As against this, the slicing only presents a view.

There are two types of advanced indexing − Integer and Boolean.

https://www.tutorialspoint.com/numpy/numpy_broadcasting.htm
The term broadcasting refers to the ability of NumPy to treat arrays of different shapes during arithmetic operations. Arithmetic operations on arrays are usually done on corresponding elements. If two arrays are of exactly the same shape, then these operations are smoothly performed.



https://pangeo.io/faq.html
maybe add this as a note
What is Big Data?

There is a lot of hype around the buzzword “Big Data” today. Some people may associate Big Data with specific software platforms (e.g. Hadoop, spark), while, for others, “big data” means specific machine learning techniques. A more useful and general definition can be found on wikipedia:

    Big data is data sets that are so voluminous and complex that traditional data processing application software are inadequate to deal with them.

By this definition, a great many datasets we regularly confront in Earth science are big data.

A good threshold for when data becomes difficult to deal with is when the volume of data exceeds your computer’s RAM. Most modern laptops have between 2 and 16 GB of RAM. High-end workstations and servers can have 1 TB (1000 GB) or RAM or more. If the dataset you are trying to analyze can’t fit in you computer’s memory, some special care is required to carry out the analysis. Data that can’t fit in RAM but can fit on your hard drive is sometimes called “medium data.”

The next threshold of difficulty is when the data can’t fit on your hard drive. Most modern laptops have between 100 GB and 4 TB of storage space on the hard drive. If you can’t fit your dataset on your internal hard drive, you can buy an external hard drive. However, at that point you might consider using a high-end server, HPC system, or cloud-based storage for your dataset. Once you have many TB of data to analyze, you are definitely in the realm of Big Data.

Some widely used datasets in geoscience exceed a PB. These datasets present an extreme challenge, a challenge that the Pangeo project aims to solve.




https://jalammar.github.io/gentle-visual-intro-to-data-analysis-python-pandas
https://github.com/jalammar/pandas-intro
https://raw.githubusercontent.com/jalammar/jalammar.github.io/master/_posts/2019-06-26-visual-numpy.md
https://github.com/jalammar/jalammar.github.io/blob/master/_posts/2019-06-26-visual-numpy.md
https://github.com/jalammar/jalammar.github.io/blob/master/_posts/2018-10-29-gentle-visual-intro-to-data-analysis-python-pandas.md
https://github.com/jalammar/jalammar.github.io/blob/master/_posts/2018-04-30-visualizing-pandas-pivoting-and-reshaping.md
https://github.com/jalammar/jalammar.github.io/tree/master/_posts

https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html

https://medium.com/writers-blokke/can-ai-really-improve-your-writing-9bc3e6440973

https://axil.github.io/a-comprehensive-guide-to-numpy-data-types.html

https://www.giters.com/axil/axil.github.io

https://github.com/satwikkansal/wtfpython#section-slippery-slopes

https://github.com/axil/beautiful-docs-reborn


https://betterprogramming.pub/numpy-illustrated-the-visual-guide-to-numpy-3b1d4976de1d
The central concept of NumPy is an n-dimensional array.

    Vectors, the 1D Arrays
    Matrices, the 2D Arrays
    3D and above
At first glance, NumPy arrays are similar to Python lists. They both serve as containers with fast item getting and setting and somewhat slower inserts and removals of elements.

    more compact, especially when there’s more than one dimension
    faster than lists when the operation can be vectorized
    slower than lists when you append elements to the end
    usually homogeneous: can only work fast with elements of one type

One way to create a NumPy array is to convert a Python list. The type will be auto-deduced from the list element types:
Be sure to feed in a homogeneous list, otherwise you’ll end up with dtype=’object’, which annihilates the speed and only leaves the syntactic sugar contained in NumPy.

NumPy arrays cannot grow the way a Python list does: No space is reserved at the end of the array to facilitate quick appends. So it is a common practice to either grow a Python list and convert it to a NumPy array when it is ready or to preallocate the necessary space with np.zeros or np.empty:

But arange is not especially good at handling floats:
This 0.1 looks like a finite decimal number to us but not to the computer: In binary, it is an infinite fraction and has to be rounded somewhere thus an error. That’s why feeding a step with fractional part to arange is generally a bad idea: You might run into an off-by-one error. You can make an end of the interval fall into a non-integer number of steps (solution1) but that reduces readability and maintainability. This is where linspace might come in handy. It is immune to rounding errors and always generates the number of elements you ask for. There’s a common gotcha with linspace, though. It counts points, not intervals, thus the last argument is always plus one to what you would normally think of. So it is 11, not 10 in the example above.

difference between random.randint and np.random.randint

There’s also a new interface for random arrays generation. It is:
– better suited for multi-threading,
– somewhat faster,
– more configurable (you can squeeze even more speed or even more quality by choosing a non-default so-called ‘bit generator’),
– able to pass two tricky synthetic tests that the old version fails.

All of the indexing methods presented above except fancy indexing are actually so-called “views”: They don’t store the data and reflect the changes in the original array if it happens to get changed after being indexed.

All of those methods including fancy indexing are mutable: They allow modification of the original array contents through assignment, as shown above. This feature breaks the habit of copying arrays by slicing them: check the figure


Also, such assignments must not change the size of the array, so tricks like
won’t work in NumPy 

Another super-useful way of getting data from NumPy arrays is boolean indexing, which allows using all kinds of logical operators:
Python “ternary” comparisons like 3<=a<=5 don’t work here.
Note that np.where with one argument returns a tuple of arrays (1-tuple in 1D case, 2-tuple in 2D case, etc), thus you need to write np.where(a>5)[0] to get np.array([5,6,7]) in the example above

As usual in Python, a//b means a div b (quotient from division), x**n means xⁿ
The same way ints are promoted to floats when adding or subtracting, scalars are promoted (aka broadcasted) to arrays:
floor rounds to -∞, ceil to +∞ and around — to the nearest integer (.5 to even)


both std and var ignore Bessel’s correction and give a biased result in the most typical use case of estimating std from a sample when the population mean is unknown. The standard approach to get a less biased estimation is to have n-1 in the denominator, which is done with ddof=1 (‘delta degrees of freedom’):
Pandas std uses Bessel’s correction by default
The effect of the Bessel’s correction quickly diminishes with increasing sample size. Also, it is not a one-size-fits-all solution, e.g. for the normal distribution ddof=1.5 is better:

Searching for an element in a vector

example for wrong implentation
https://github.com/numpy/numpy/issues/10161

 I’ll use the words matrix and 2D array interchangeably.
Double parentheses are necessary here because the second positional parameter is reserved for the (optional) dtype (which also accepts integers).

The “view” sign means that no copying is actually done when slicing an array. 
The 2D case is somewhat counter-intuitive: you need to specify the dimension to be eliminated, instead of the remaining one you would normally think about. 

ordinary operators (like +,-,*,/,// and **) which work element-wise, there’s a @ operator that calculates a matrix product:
Note that in the last example it is a symmetric per-element multiplication. To calculate the outer product using an asymmetric linear algebra matrix multiplication the order of the operands should be reversed:

None in the square brackets serves as a shortcut for np.newaxis, which adds an empty axis at the designated place.

flatten is always a copy, reshape(-1) is always a view, ravel is a view when possible

By the rules of broadcasting, 1D arrays are implicitly interpreted as 2D row vectors,

Strictly speaking, any array, all but one dimensions of which are single-sized, is a vector (eg. a.shape==[1,1,1,5,1,1]), so there’s an infinite number of vector types in numpy, but only these three are commonly used. You can use np.reshape to convert a ‘normal’ 1D vector to this form and np.squeeze to get it back. Both functions act as views.

Those two work fine with stacking matrices only or vectors only, but when it comes to mixed stacking of 1D arrays and matrices, only the vstack works as expected: The hstack generates a dimensions-mismatch error because as described above, the 1D array is interpreted as a row vector, not a column vector. The workaround is either to convert it to a row vector or to use a specialized column_stack function which does it automatically:


Actually, if all you need to do is add constant values to the border(s) of the array, the (slightly overcomplicated) pad function should suffice:

The meshgrid function accepts an arbitrary set of indices, mgrid — just slices and indices can only generate the complete index ranges. fromfunction calls the provided function just once, with the I and J argument as described above.
But actually, there is a better way to do it in NumPy

Here flipud flips the matrix in the up-down direction (to be precise, in the axis=0 direction, same as a[::-1,...], where three dots mean “all other dimensions”—



https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises_with_hints_with_solutions.md
26. What is the output of the following script? (★☆☆)
34. How to get all the dates corresponding to the month of July 2016? (★★☆)
35. How to compute ((A+B)*(-A/2)) in place (without copy)? (★★☆)
41. How to sum a small array faster than np.sum? (★★☆)
43. Make an array immutable (read-only) (★★☆)
49. How to print all the values of an array? (★★☆)
71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★)
68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset indices? (★★★)
72. How to swap two rows of an array? (★★★)
75. How to compute averages using a sliding window over an array? (★★★)
81. Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]? (★★★)
93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★)


https://numpy.org/
NumPy offers comprehensive mathematical functions, random number generators, linear algebra routines, Fourier transforms, and more.
The core of NumPy is well-optimized C code. Enjoy the flexibility of Python with the speed of compiled code.


https://docs.scipy.org/doc/scipy-0.15.1/reference/tutorial/general.html
The additional benefit of basing SciPy on Python is that this also makes a powerful programming language available for use in developing sophisticated programs and specialized applications. Scientific applications using SciPy benefit from the development of additional modules in numerous niche’s of the software landscape by developers across the world. Everything from parallel programming to web and data-base subroutines and classes have been made available to the Python programmer. All of this power is available in addition to the mathematical libraries in SciPy.

This tutorial will acquaint the first-time user of SciPy with some of its most important features. It assumes that the user has already installed the SciPy package. Some general Python facility is also assumed, such as could be acquired by working through the Python distribution’s Tutorial. For further introductory help the user is directed to the Numpy documentation.

For brevity and convenience, we will often assume that the main packages (numpy, scipy, and matplotlib) have been imported as:
>>>

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

These are the import conventions that our community has adopted after discussion on public mailing lists. You will see these conventions used throughout NumPy and SciPy source code and documentation. While we obviously don’t require you to follow these conventions in your own code, it is highly recommended.

.. _numpy:

numpy
-----


     


.. _pandas:

pandas
------

.. _scipy:

scipy
-----





https://axil.github.io/a-comprehensive-guide-to-numpy-data-types.html
Pandas has a separate data type for that, but NumPy's way of dealing with the missed values is through the so-called masked array: you mark the invalid values with a boolean mask and then all the operations are carried out as if the values are not there.

>>> np.array([4,0,6]).mean()          # the value 0 means 'missing' here
3.3333333333333335
>>> import numpy.ma as ma
>>> ma.array([4,0,6], mask=[0,1,0]).mean()
5.0




vectorization
https://nbviewer.org/github/pydata/pydata-book/blob/2nd-edition/appa.ipynb
Writing New ufuncs in Python

def add_elements(x, y):
    return x + y
add_them = np.frompyfunc(add_elements, 2, 1)
add_them(np.arange(8), np.arange(8))

add_them = np.vectorize(add_elements, otypes=[np.float64])
add_them(np.arange(8), np.arange(8))

arr = np.random.randn(10000)
%timeit add_them(arr, arr)
%timeit np.add(arr, arr)





https://git.ichec.ie/training/studentresources/hpc-python/march-2021/-/blob/master/01-NumPy/01-NumPy.ipynb
By default, numexpr tries to use multiple threads.
Number of threads can be queried and set with ne.set_num_threads(nthreads)



svd with dask
https://examples.dask.org/machine-learning/svd.html

https://examples.dask.org/applications/stencils-with-numba.html
Comparing Stencil Computations on the CPU and GPU
https://gist.github.com/mrocklin/9272bf84a8faffdbbe2cd44b4bc4ce3c





https://aaltoscicomp.github.io/data-analysis-workflows-course/chapter-4-scaling/#parallelization-strategies
