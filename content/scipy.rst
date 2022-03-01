SciPy
=====

.. questions::

   - 1

.. objectives::

   - 1
   - 2 

.. seealso::

   * Main article: `SciPy documentation <https://docs.scipy.org/doc/scipy/reference/>`__



SciPy is a library that builds on top of NumPy. It contains a lot of
interfaces to battle-tested numerical routines written in Fortran or
C, as well as python implementations of many common algorithms.



What's in SciPy?
----------------

Briefly, it contains functionality for

- Special functions (Bessel, Gamma, etc.)
- Numerical integration
- Optimization
- Interpolation
- Fast Fourier Transform (FFT)
- Signal processing
- Linear algebra (more complete than in NumPy)
- Sparse matrices
- Statistics
- More I/O routine, e.g. Matrix Market format for sparse matrices,
  MATLAB files (.mat), etc.

Many (most?) of these are not written specifically for SciPy, but use
the best available open source C or Fortran libraries.  Thus, you get
the best of Python and the best of compiled languages.

Most functions are documented ridiculously well from a scientific
standpoint: you aren't just using some unknown function, but have a
full scientific description and citation to the method and
implementation.



.. keypoints::

   - When you need advance math or scientific functions, let's just
     admit it: you do a web search first.
   - But when you see something in SciPy come up, you know your
     solutions are in good hands.
