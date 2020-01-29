.. Copied from gpytorch.
:github_url: https://github.com/CedricTravelletti/Volcano

Volcapy: Bayesian Inversion for large scale Geophysical Inverse Problems (and beyond ...)
=========================================================================================

The Bayesian approach to inverse problem is a well-known and powerful one :cite:`tarantola`,
:cite:`stuart_2010`

This module implements the set uncetainty quantification methods proposed in
:cite:`chevalier_uq`, :cite:`azzimonti_uq`, :cite:`azzimonti_adaptive`.

The main goal is to identify regions in model space where the density field
might be above some given threshold, using the posterior distribution.
We call such regions **excursion set** above the threshold.
We also aim at quantifying the uncertainty on the estimated regions.

.. math::

   \Gamma = \lbrace x \in X: \tilde{Z}_x \geq u_0 \rbrace

Module Functionalities
~~~~~~~~~~~~~~~~~~~~~~

========================================== ======================================================================================
Excursion Set Methods
=================================================================================================================================
Out-of-core matrix-matrix multiplication    Multiply matrices that do not fit in memory.
GPU Inversion                               Solve inverse problems on multiple GPUs. 
Hyperparameter Optimization                 Fit model hyperparameters using maximum likelihood.
Set Estimation                              Get cells belonging to the Vorob'ev quantile at a given level, for a given threshold.
========================================== ======================================================================================


.. bibliography:: bibliography.bib



.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Tutorials:

   examples/tuto.ipynb
   examples/README.md

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Main Modules:

   inverse
   gaussian_process
   covariance
   synthetic
   uq
   update

.. toctree::
   :maxdepth: 1
   :caption: Advanced Usage

   train
   plotting
   niklas
   forward
   grid
   compatibility_layer


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
