Optimization
============

Once your code is working reliably, you can start thinking of optimizing it.


.. warning::

Always measure the code before you start optimization. Don't base your optimization 
on theoretical consideration, otherwise you'll have surprises. 


Profilers 
---------

Timeit
******

If you use Jupyter-notebook, the best choice will be to use ``timeit`` (https://docs.python.org/library/timeit.html) to time a small piece of code:

.. sourcecode:: ipython

    In [1]: import numpy as np

    In [2]: a = np.arange(1000)

    In [3]: %timeit a ** 2
    100000 loops, best of 3: 5.73 us per loop

.. note::

   For long running calls, using ``%time`` instead of ``%timeit``; it is
   less precise but faster


cProfile
********

For more complex code, one could use the built-in python profilers 
<https://docs.python.org/3/library/profile.html>`_ ``cProfile``.

    .. sourcecode:: console

        $  python -m cProfile -o demo.prof demo.py

    Using the ``-o`` switch will output the profiler results to the file
    ``demo.prof`` to view with an external tool. This can be useful if
    you wish to process the profiler output with a visualization tool.


Line-profiler
*************

The cprofile tells us which function takes most of the time, but not where it is called.

For this information, we use the `line_profiler <http://packages.python.org/line_profiler/>`_: in the
source file  by adding a decorator ``@profile`` in the functions of interests

.. sourcecode:: python

    @profile
    def test():
        data = np.random.random((5000, 100))
        u, s, v = linalg.svd(data)
        pca = np.dot(u[:, :10], data)
        results = fastica(pca.T, whiten=False)

Then we run the script using the `kernprof.py
<http://packages.python.org/line_profiler>`_ program, with switches ``-l, --line-by-line`` and ``-v, --view`` to use the line-by-line profiler and view the results in addition to saving them:

.. sourcecode:: console

    $ kernprof.py -l -v demo.py

    Wrote profile results to demo.py.lprof
    Timer unit: 1e-06 s

    File: demo.py
    Function: test at line 5
    Total time: 14.2793 s

    Line #      Hits         Time  Per Hit   % Time  Line Contents
    =========== ============ ===== ========= ======= ==== ========
        5                                           @profile
        6                                           def test():
        7         1        19015  19015.0      0.1      data = np.random.random((5000, 100))
        8         1     14242163 14242163.0   99.7      u, s, v = linalg.svd(data)
        9         1        10282  10282.0      0.1      pca = np.dot(u[:10, :], data)
       10         1         7799   7799.0      0.1      results = fastica(pca.T, whiten=False)

**The SVD is taking all the time.** We need to optimise this line.



performance enhancement 
-----------------------

Once we have identified the bottlenecks, we need to make the corresponding code go faster.

Algorithmic optimization
************************

The first thing to look into is the underlying algorithm you chose: is it optimal?
To answer this question,  a good understanding of the maths behind the algorithm helps. 
However, it can be as simple as moving computation or memory allocation outside a loop, and this happens very often.

SVD
...................

SVD `Singular Value Decomposition <https://en.wikipedia.org/wiki/Singular_value_decomposition>`_
is quite often used in climate model data analysis.  The computational cost of this algorithm is 
roughly :math:`n^3` in the size of the input matrix. 
However, in both of these example, we are not using all the output of
the SVD, but only the first few rows of its first return argument. If
we use the ``svd`` implementation of scipy, we can ask for an incomplete
version of the SVD. Note that implementations of linear algebra in
scipy are richer then those in numpy and should be preferred.

.. sourcecode:: ipython

    In [3]: %timeit np.linalg.svd(data)
    1 loops, best of 3: 14.5 s per loop

    In [4]: from scipy import linalg

    In [5]: %timeit linalg.svd(data)
    1 loops, best of 3: 14.2 s per loop

    In [6]: %timeit linalg.svd(data, full_matrices=False)
    1 loops, best of 3: 295 ms per loop

    In [7]: %timeit np.linalg.svd(data, full_matrices=False)
    1 loops, best of 3: 293 ms per loop

We can then use this insight to :download:`optimize the previous code <demo_opt.py>`:

.. literalinclude:: demo_opt.py
   :pyobject: test

.. sourcecode:: ipython

    In [1]: import demo

    In [2]: %timeit demo.
    demo.fastica   demo.np        demo.prof.pdf  demo.py        demo.pyc
    demo.linalg    demo.prof      demo.prof.png  demo.py.lprof  demo.test

    In [2]: %timeit demo.test()
    ica.py:65: RuntimeWarning: invalid value encountered in sqrt
      W = (u * np.diag(1.0/np.sqrt(s)) * u.T) * W  # W = (W * W.T) ^{-1/2} * W
    1 loops, best of 3: 17.5 s per loop

    In [3]: import demo_opt

    In [4]: %timeit demo_opt.test()
    1 loops, best of 3: 208 ms per loop

Real incomplete SVDs, e.g. computing only the first 10 eigenvectors, can
be computed with arpack, available in ``scipy.sparse.linalg.eigsh``.

.. topic:: Computational linear algebra

    For certain algorithms, many of the bottlenecks will be linear
    algebra computations. In this case, using the right function to solve
    the right problem is key. For instance, an eigenvalue problem with a
    symmetric matrix is easier to solve than with a general matrix. Also,
    most often, you can avoid inverting a matrix and use a less costly
    (and more numerically stable) operation.

add sparse matrix here 

CPU usage optimization
************************

Vectorization
..................

Arithmetic is one place where numpy performance outperforms python list and the reason is that it uses vectorization.
A lot of the data analysis involves a simple operation being applied to each element of a large dataset.
In such cases, vectorization is key for better performance.

.. challenge:: scalar vector multiplication 

   .. tabs::

      .. tab:: python

             .. code-block:: python

			a = [1, 3, 5]
			b = 10 *a 

      .. tab:: numpy

             .. code-block:: python

			import numpy as np
                        a = np.array([1, 3, 5])
                        b = 10 *a 








Memory usage optimization
*************************

Broadcasting
............

Basic operations of numpy are elementwise, and the shape of the arrays should be compatible.
However, in practice under certain conditions, it is possible to do operations on arrays of different shapes.
NumPy expands the arrays such that the operation becomes viable.

.. note::

Broadcasting Rules
Dimensions match when they are equal, or when either is 1 or None. 
In the latter case, the dimension of the output array is expanded to the larger of the two.

   .. tabs:: broadcasting

      .. tab:: 1D

             .. code-block:: python

			     a = np.array([[1, 2, 3],
	                	   [4, 5, 6]])
			     b = np.array([10, 10, 10])
			     a + b                       # array([[11, 12, 13],
                                			 #        [14, 15, 16]]) 


      .. tab:: 2D

             .. code-block:: 2D

			import numpy as np
                        a = np.array([1, 3, 5])
                        b = 10 *a 




.. figure:: https://numpy.org/doc/stable/_images/broadcasting_1.png

   Source: `numpy.org <https://numpy.org/doc/stable/_images/broadcasting_1.png>`__.

.. figure:: https://numpy.org/doc/stable/_images/broadcasting_2.png

   Source: `numpy.org <https://numpy.org/doc/stable/_images/broadcasting_2.png>`__.

.. figure:: https://numpy.org/doc/stable/_images/broadcasting_3.png

   Source: `numpy.org <https://numpy.org/doc/stable/_images/broadcasting_3.png>`__.



.. figure:: https://numpy.org/doc/stable/_images/broadcasting_4.png

   Source: `numpy.org <https://numpy.org/doc/stable/_images/broadcasting_4.png>`__.


.. figure:: https://scipy-lectures.org/_images/numpy_broadcasting.png

   Source: `scipy-lectures.org <https://scipy-lectures.org/_images/numpy_broadcasting.png>`__.



.. note::

the broadcasted arrays are never physically constructed


   .. tabs:: broadcasting

      .. tab:: 1D

             .. figure:: img/broadcasting_1.png 


      .. tab:: 2D

             .. figure:: img/broadcasting_2.png 

		



Cash
............



In place operations
............


  .. sourcecode:: ipython

    In [1]: a = np.zeros(1e7)

    In [2]: %timeit global a ; a = 0*a
    10 loops, best of 3: 111 ms per loop

    In [3]: %timeit global a ; a *= 0
    10 loops, best of 3: 48.4 ms per loop

  **note**: we need `global a` in the timeit so that it work, as it is
  assigning to `a`, and thus considers it as a local variable.

* **Be easy on the memory: use views, and not copies**

  Copying big arrays is as costly as making simple numerical operations
  on them:

  .. sourcecode:: ipython

    In [1]: a = np.zeros(1e7)

    In [2]: %timeit a.copy()
    10 loops, best of 3: 124 ms per loop

    In [3]: %timeit a + 1
    10 loops, best of 3: 112 ms per loop

* **Beware of cache effects**

  Memory access is cheaper when it is grouped: accessing a big array in a
  continuous way is much faster than random access. This implies amongst
  other things that **smaller strides are faster** (see
  :ref:`cache_effects`):

  .. sourcecode:: ipython

    In [1]: c = np.zeros((1e4, 1e4), order='C')

    In [2]: %timeit c.sum(axis=0)
    1 loops, best of 3: 3.89 s per loop

    In [3]: %timeit c.sum(axis=1)
    1 loops, best of 3: 188 ms per loop

    In [4]: c.strides
    Out[4]: (80000, 8)

  This is the reason why Fortran ordering or C ordering may make a big
  difference on operations:

  .. sourcecode:: ipython

    In [5]: a = np.random.rand(20, 2**18)

    In [6]: b = np.random.rand(20, 2**18)

    In [7]: %timeit np.dot(b, a.T)
    1 loops, best of 3: 194 ms per loop

    In [8]: c = np.ascontiguousarray(a.T)

    In [9]: %timeit np.dot(b, c)
    10 loops, best of 3: 84.2 ms per loop

  Note that copying the data to work around this effect may not be worth it:

  .. sourcecode:: ipython

    In [10]: %timeit c = np.ascontiguousarray(a.T)
    10 loops, best of 3: 106 ms per loop

  Using `numexpr <http://code.google.com/p/numexpr/>`_ can be useful to
  automatically optimize code for such effects.

compiled code
*************************

For many use cases using NumPy or Pandas is sufficient. Howevewr, in some computationally heavy applications, 
it is possible to improve the performance by using compiled code.
Normally Cython and Numba are among the popular choices and both of them have good support for numpy arrays. 


cython
.......

The source code gets translated into optimized C/C++ code and compiled as Python extension modules. 


numba
.......


An alternative to statically compiling Cython code is to use a dynamic just-in-time (JIT) compiler with `Numba <https://numba.pydata.org/>`__.

Numba allows you to write a pure Python function which can be JIT compiled to native machine instructions, similar in performance to C, C++ and Fortran, by decorating your function with ``@jit``.

Numba works by generating optimized machine code using the LLVM compiler infrastructure at import time, runtime, or statically (using the included pycc tool).
Numba supports compilation of Python to run on either CPU or GPU hardware and is designed to integrate with the Python scientific software stack.

.. note::

    The ``@jit`` compilation will add overhead to the runtime of the function, so performance benefits may not be realized especially when using small data sets.
    Consider `caching <https://numba.readthedocs.io/en/stable/developer/caching.html>`__ your function to avoid compilation overhead each time your function is run.

Numba can be used in 2 ways with pandas:

#. Specify the ``engine="numba"`` keyword in select pandas methods
#. Define your own Python function decorated with ``@jit`` and pass the underlying NumPy array of :class:`Series` or :class:`DataFrame` (using ``to_numpy()``) into the function

If Numba is installed, one can specify ``engine="numba"`` in select pandas methods to execute the method using Numba.
Methods that support ``engine="numba"`` will also have an ``engine_kwargs`` keyword that accepts a dictionary that allows one to specify
``"nogil"``, ``"nopython"`` and ``"parallel"`` keys with boolean values to pass into the ``@jit`` decorator.
If ``engine_kwargs`` is not specified, it defaults to ``{"nogil": False, "nopython": True, "parallel": False}`` unless otherwise specified.

In terms of performance, **the first time a function is run using the Numba engine will be slow**
as Numba will have some function compilation overhead. However, the JIT compiled functions are cached,
and subsequent calls will be fast. In general, the Numba engine is performant with
a larger amount of data points (e.g. 1+ million).





Consider the following pure Python code:

.. literalinclude:: example/integrate.py

Simply compiling this in Cython merely gives a 35% speedup.  This is
better than nothing, but adding some static types can make a much larger
difference.


   .. tabs:: 

      .. tab:: python

             .. literalinclude:: example/integrate.py 

      .. tab:: numpy

              .. literalinclude:: example/integrate.numpy.py 

      .. tab:: cython

              .. literalinclude:: example/integrate.cython.py 

      .. tab:: numba

              .. literalinclude:: example/integrate.numba.py 
