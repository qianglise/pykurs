.. _dask:

Dask for scalable analytics
===========================

.. objectives::

   - Understand how Dask achieves parallelism
   - Learn a few common workflows with Dask


Overview
--------

An increasingly common problem faced by researchers and data scientists 
today is that datasets are becoming larger and larger and modern data analysis 
is thus becoming more and more computationally demanding. The first 
difficulty to deal with is when the volume of data exceeds one's computer's RAM. 
Modern laptops/desktops have about 10 GB of RAM. Beyond this threshold, 
some special care is required to carry out data analysis. 
The next threshold of difficulty is when the data can not even 
fit on the hard drive, which is about a couple of TB on a modern laptop.
In this situation, it is better to use an HPC system or a cloud-based solution, 
and Dask is a tool that helps us easily extend our familiar data analysis 
tools to work with big data. In addition, Dask can also speeds up 
our analysis by using multiple CPU cores which makes our work run 
faster on laptop, HPC and cloud platforms.

What is Dask?
-------------

Dask is composed of two parts:

- Dynamic task scheduling optimized for computation. Similar to other workflow 
  management systems, but optimized for interactive computational workloads.
- "Big Data" collections like parallel arrays, dataframes, and lists that extend 
  common interfaces like NumPy, Pandas, or Python iterators to larger-than-memory 
  or distributed environments. These parallel collections run on top of dynamic 
  task schedulers.

.. figure:: img/dask-overview.svg

   High level collections are used to generate task graphs which can be executed 
   by schedulers on a single machine or a cluster. From the 
   `Dask documentation <https://docs.dask.org/en/stable/>`__.

Dask Clusters
-------------

Dask needs computing resources in order to perform parallel computations. 
"Dask Clusters" have different names corresponding to different computing environments, 
for example: 

  - `LocalCluster` on laptop/desktop
  - `PBSCluster` or SLURMCluster on HPC
  - `Kubernetes` cluster in the cloud
 
Each cluster will be allocated with a given number of "workers" associated with 
CPU and RAM and the Dask scheduling system automatically maps jobs to each worker.

Here we will focus on using a LocalCluster, and it is recommended to use 
a distributed sceduler ``dask.distributed``. It is more sophisticated, offers more features,
but requires minimum effort to set up. It can run locally on a laptop and scale up to a cluster. 
We can start a LocalCluster scheduler which makes use of all the cores and RAM 
we have on the machine by: 

.. code-block:: python
    
    from dask.distributed import Client, LocalCluster
    # create a local cluster
    cluster = LocalCluster()
    # connect to the cluster we just created
    client = Client(cluster)
    client


Or you can simply lauch a Client() call which is shorthand for what is described above.

.. code-block:: python

    from dask.distributed import Client
    client = Client()
    client


We can also specify the resources to be allocated to a Dask cluster by:

.. code-block:: python
    
    from dask.distributed import Client, LocalCluster
    # create a local cluster with 
    # 4 workers 
    # 1 thread per worker
    # 4 GiB memory limit for a worker
    cluster = LocalCluster(n_workers=4,threads_per_worker=1,memory_limit='4GiB')


Cluster managers also provide useful utilities: for example if a cluster manager supports scaling, 
you can modify the number of workers manually or automatically based on workload:

.. code-block:: python
   
   cluster.scale(10)  # Sets the number of workers to 10
   cluster.adapt(minimum=1, maximum=10)  # Allows the cluster to auto scale to 10 when tasks are computed


Dask distributed scheduler also provides live feedback via its interactive dashboard. 
A link that redirects to the dashboard will prompt in the terminal 
where the scheduler is created, and it is also shown when you create a Client and connect the scheduler.
By default, when starting a scheduler on your local machine the dashboard will be served at 
http://localhost:8787/status and can be always queried from commond line by:

.. code-block:: python

   cluster.dashboard_link 
   http://127.0.0.1:8787/status
   # or 
   client.dashboard_link



When everything finishes, you can shut down the connected scheduler and workers 
by calling the :meth:`shutdown` method:

.. code-block:: python

   client.shutdown()




Dask Collections
----------------

Dask provides dynamic parallel task scheduling and 
three main high-level collections:
  
  - ``dask.array``: Parallel NumPy arrays
  - ``dask.dataframe``: Parallel Pandas DataFrames
  - ``dask.bag``: Parallel Python Lists 


Dask Arrays
^^^^^^^^^^^

A Dask array looks and feels a lot like a NumPy array. 
However, a Dask array uses the so-called "lazy" execution mode, 
which allows one to build up complex, large calculations symbolically 
before turning them over the scheduler for execution. 


.. callout:: Lazy evaluation

   Contrary to normal computation, lazy execution mode is when all the computations 
   needed to generate results are symbolically represented, forming a queue of 
   tasks mapped over data blocks. Nothing is actually computed until the actual 
   numerical values are needed, e.g., to print results to your screen or write to disk. 
   At that point, data is loaded into memory and computation proceeds in a streaming 
   fashion, block-by-block. The actual computation is controlled by a multi-processing 
   or thread pool, which allows Dask to take full advantage of multiple processors 
   available on the computers.


.. code-block:: python

    import numpy as np
    shape = (1000, 4000)
    ones_np = np.ones(shape)
    ones_np
    ones_np.nbytes / 1e6


Now let's create the same array using Dask's array interface. In addition to 
providing the shape of the array, we also specify the ``chunks`` argument, 
which describes how the array is split up into sub-arrays:

.. code-block:: python

    import dask.array as da
    shape = (4000, 4000)
    chunk_shape = (1000, 1000)
    ones = da.ones(shape, chunks=chunk_shape)
    ones

.. note::Other ways to specify ``chunks`` size can be found here https://docs.dask.org/en/stable/array-chunks.html#specifying-chunk-shapes


So far, it is only a symbolic representation of the array. 
One way to trigger the computation is to call :meth:`compute`:

.. code-block:: python

    ones.compute()


.. note::

   Plotting also triggers computation, since the actual values are needed to produce the plot.


We can visualize the symbolic operations by calling :meth:`visualize`:

.. code-block:: python

    ones.visualize()

Let us calculate the sum of the dask array and visualize again:

.. code-block:: python

    sum_da = ones.sum()
    sum_da.visualize()

You can find additional details and examples here 
https://examples.dask.org/array.html.

Dask Dataframe
^^^^^^^^^^^^^^

Dask dataframes split a dataframe into partitions along an index and can be used 
in situations where one would normally use Pandas, but this fails due to data size or 
insufficient computational efficiency. Specifically, you can use Dask dataframes to:

- manipulate large datasets, even when these don't fit in memory
- accelerate long computations by using many cores
- perform distributed computing on large datasets with standard Pandas operations 
  like groupby, join, and time series computations.

Let us revisit the dataset containing the Titanic passenger list, and now transform it to 
a Dask dataframe:

.. code-block:: python

   import pandas as pd
   import dask.dataframe as dd

   url = "https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/titanic.csv"
   df = pd.read_csv(url, index_col="Name")

   ddf = dd.from_pandas(df, npartitions=10)


Dask dataframes do not support the entire interface of Pandas dataframes, but 
the most `commonly used methods are available <https://docs.dask.org/en/stable/dataframe.html#scope>`__. 
For a full listing refer to the 
`dask dataframe API <https://docs.dask.org/en/stable/dataframe-api.html>`__.

We can for example perform the group-by operation we did earlier, but this time in parallel:

.. code-block:: python

   ddf[ddf["Age"] < 12].groupby(["Sex", "Child"])["Survived"].mean().compute()

However, for a small dataframe like this the overhead of parallelisation will far 
outweigh the benefit. 

As an additional use case, recall the word-count project that we encountered earlier. 
The :download:`results.txt <data/results.txt>` file contains word counts of the 10 
most frequent words in different texts, and we want to fit a power law to the 
individual distributions in each row.

Here is our fitting function:

.. code-block:: python

   def linear_fit_loglog(row):
       X = np.log(np.arange(row.shape[0]) + 1.0)
       ones = np.ones(row.shape[0])
       A = np.vstack((X, ones)).T
       Y = np.log(row)
       res = np.linalg.lstsq(A, Y, rcond=-1)
       return res[0][0]

Earlier we saw that iterating over a pandas dataframe was slower than using the 
:meth:`apply` function. With dask dataframes, we should not iterate over dataframes at all!
We load the `results.txt` file directly into a dask dataframe and fit the power law 
to each row:

.. code-block:: python

   ddf = dd.read_csv("/some/path/to/results.txt")
   results = ddf.iloc[:,1:].apply(linear_fit_loglog, axis=1, meta=(None, "float64"))

Note the additional argument ``meta`` which is required for dask dataframes. 
It should contain an empty ``pd.DataFrame`` or ``pd.Series`` that matches the 
dtypes and column names of the output, or a dict of ``{name: dtype}`` or iterable of ``(name, dtype)``. 

You can find additional details and examples here 
https://examples.dask.org/dataframe.html.


Dask Bag
^^^^^^^^

A Dask bag enables processing data that can be represented as a sequence of arbitrary 
inputs ("messy data"), like in a Python list. Dask Bags are often used to for 
preprocessing log files, JSON records, or other user defined Python objects.

We will content ourselves with implementing a dask version of the word-count problem, 
specifically the step where we count words in a text. 

.. type-along:: Dask version of word-count

   First navigate to the ``word-count-hpda`` directory. The serial version (wrapped in 
   multiple functions in the ``source/wordcount.py`` code) looks like this:

   .. code-block:: python

      filename = './data/pg10.txt'
      DELIMITERS = ". , ; : ? $ @ ^ < > # % ` ! * - = ( ) [ ] { } / \" '".split()
      
      with open(filename, "r") as input_fd:
          lines = input_fd.read().splitlines()
      
      counts = {}
      for line in lines:
          for purge in DELIMITERS:
              line = line.replace(purge, " ")
          words = line.split()
          for word in words:
              word = word.lower().strip()
              if word in counts:
                  counts[word] += 1
              else:
                  counts[word] = 1    
      
      sorted_counts = sorted(list(counts.items()), key=lambda key_value: key_value[1], reverse=True)
      
      sorted_counts[:10]

   A very compact ``dask.bag`` version of this code is as follows:

   .. code-block:: python

      import dask.bag as db
      filename = './data/pg10.txt'
      DELIMITERS = ". , ; : ? $ @ ^ < > # % ` ! * - = ( ) [ ] { } / \" '".split()

      text = db.read_text(filename, blocksize='1MiB')
      sorted_counts = text.filter(lambda word: word not in DELIMITERS).str.lower().str.strip().str.split().flatten().frequencies().topk(10,key=1).compute()

      sorted_counts

.. callout:: When to use a Dask bag

   There is no benefit from using a Dask bag on small datasets. But imagine we were 
   analysing a very large text file (all tweets in a year? a genome?). Dask provides 
   both parallelisation and the ability to utilize RAM on multiple machines.



Dask Delayed
^^^^^^^^^^^^

Sometimes problems don't fit into one of the collections like 
``dask.array`` or ``dask.dataframe``. In these cases, we can parallelise custom algorithms 
using ``dask.delayed`` interface. ``dask.delayed`` allows users to delay function calls 
into a task graph with dependencies. If you have a problem that is paralellisable, 
but isn't as simple as just a big array or a big dataframe, then ``dask.delayed`` 
may be the right choice.


Consider the following example. The functions are very simple, and they sleep 
for a prescribed time to simulate real work.

.. literalinclude:: example/delay.py 
   :language: python

Let us run the example first, one after the other in sequence:

.. sourcecode:: ipython

    %%time
    x = inc(1)
    y = dec(2)
    z = add(x, y)
    z


Note that the first two functions ``inc`` and ``dec`` don't depend on each other, 
we could have called them in parallel. We can call ``dask.delayed`` on these funtions 
to make them lazy and tasks into a graph which we will run later on parallel hardware.

.. sourcecode:: ipython

    import dask
    inc = dask.delayed(inc)
    dec = dask.delayed(dec)
    add = dask.delayed(add)

    %%time
    x = inc(1)
    y = dec(2)
    z = add(x, y)
    z

    z.visualize(rankdir='LR')

    %%time
    z.compute()


Let us extend the example a little bit more by 
applying the function on a data array using for loop:

.. code-block:: ipython

    def inc(x):
        time.sleep(4)
        return x + 1

    def dec(x):
        time.sleep(3)
        return x - 1

    def add(x, y):
        time.sleep(1)
        return x + y

    data = [1, 2, 3, 4, 5]

    output = []
    for x in data:
        a = inc(x)
        b = dec(x)
        c = add(a, b)
        output.append(c)

    total = sum(output)



.. challenge:: chunk size

    The following example calculate the mean value of a ramdom generated array. 
    Run the example and see the performance improvement by using dask.
    But what happens if we use different chunk sizes?

    - Try out with different chunk sizes:
      What happens if the dask chunks=(20000,20000)
      What happens if the dask chunks=(200,200)

   .. tabs::

      .. tab:: numpy

         .. literalinclude:: example/chunk_np.py
            :language: python

      .. tab:: dask

         .. literalinclude:: example/chunk_dask.py
            :language: python



.. challenge:: Data from climate simulation

    There are a couple of data in NetCDF files containing monthly global 2m air temperature. 


.. code-block:: python

    ds=xr.open_mfdataset('/home/x_qiali/qiang/hpda/airdata/tas*.nc', parallel=True)
    ds
    ds.tas
    dask.visualize(ds.tas) 
    
    tas_mean=ds.tas.mean(axis=0) 
    fig = plt.figure
    plt.imshow(tas_mean, cmap='RdBu_r');



SVD 
---

We can use dask to compute SVD of certain matrix.

.. code-block:: python

    import dask
    import dask.array as da
    X = da.random.random((200000, 100), chunks=(10000, 100))
    u, s, v = da.linalg.svd(X)
    dask.visualize(u, s, v)


We could also use approximate algorithm

.. code-block:: python

    import dask
    import dask.array as da
    X = da.random.random((10000, 10000), chunks=(2000, 2000)).persist()
    u, s, v = da.linalg.svd_compressed(X, k=5)
    dask.visualize(u, s, v)





How does Dask work?
-------------------


Common use cases
----------------



Comparison to Spark
-------------------

Dask has much in common with the 
[Apache Spark](https://spark.apache.org/).

- ref: https://docs.dask.org/en/stable/spark.html



Exercises
---------

.. exercise:: Benchmarking dask.dataframes.apply()

   Compare the performance of :meth:`dask.dataframes.apply` with :meth:`pandas.dataframes.apply` 
   for the word-count example. You will probably see a slowdown due to the parallelisation 
   overhead. 
   But what if you add a ``time.sleep(0.01)`` inside ``linear_fit_loglog`` to 
   emulate a time-consuming calculation? 

.. exercise:: Break down the dask.bag computational pipeline

   Revisit the word-count problem and the implementation with a ``dask.bag`` that we 
   saw above. 
   
   - To get a feeling for the computational pipeline, break down the computation into 
     separate steps and investigate intermediate results using :meth:`.compute`.
   - Benchmark the serial and ``dask.bag`` versions. Do you see any speedup? 
     What if you have a larger textfile? You can for example concatenate all texts into 
     a single file: ``cat data/*.txt > data/all.txt``.




.. keypoints::

   - 1
   - 2
   - 3
