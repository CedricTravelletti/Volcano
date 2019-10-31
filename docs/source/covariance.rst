.. role:: hidden
    :class: hidden-section

volcapy.covariance
===================================

This package implements the different covariance kernels that one can use
in volcapy.

Its main goal is to compute the covariance pushforward :math:`K F^t`, where
:math:`K` is the model covariance matrix and :math:`F` is the forward operator.

**IMPORTANT**: Note that we always strip the variance parameter :math:`\sigma_0^2` from the
covariance matrix. Hence, when using the covariance pushforward computed here,
one has to manually multiply by :math:`\sigma_0^2` for expressions to make sense.


Each of the kernels should implement the three methods below

.. code-block:: python

    def compute_cov_pushforward(lambda0, F, cells_coords, device,
            n_chunks=200, n_flush=50):
        """ Compute covariance pushforward

        """
    
    def compute_cov(lambda0, cells_coords, i, j):
        """ Compute the covariance bewtween cells i and j of the model.

        """

    def compute_full_cov(lambda0, cells_coords, device,
            n_chunks=200, n_flush=50):
        """ Compute the full covariance matrix. Note that due to the
        :math:`n_m^2` size, this should only be
        attempted on small models.

        """

A detailed description of the arguments is available at the end of this
section.


Handling out of Memory Errors
-----------------------------
Due to the size of the covariance matrix, care has to be taken when computing
its product with the forward. Let :math:`n_m` be the number of model cells.
Then the covariance matrix has size :math:`n_m^2`, which for 10000 cels already
takes more than 160 Gb of memory.

The strategy used here is to compute the matrix in chunks. We compute matrix
products of the form :math:`K A` by computing the rows of the resulting matrix
in chunks of size :code:`n_chunks`. This then only involves :code:`n_chunks` of
the covariance matrix at a time.
Hence what we do is compute such a chunk of the covariance matrix on GPU,
multiply it with the right hand side matrix and send the result back to CPU
where it is concatenated with the previously computed chunks, while the freed
GPU memory is used to compute the next chunk.


We noticed that CUDA tends to keep arbitrary data in cache, which after
computing a certain number of chunks will fill the GPU memory. The cache thus
has to be manually flushed every :code`n_flush` chunks.

Flushing takes a long time, so one shouldn't do it to often. The value of
:code:`n_flush` should be as high as possible to avoid flushing too often. The
optimal value should be determined experimentally by the user.


Matérn 3/2
------------------
The implementation of the Matérn 3/2 kernel is provided as example below.

.. automodule:: volcapy.covariance.matern32
   :members:
