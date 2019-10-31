.. role:: hidden
    :class: hidden-section

Uncertainty Quantification Tools
================================

This module implements the set uncetainty quantification methods proposed in
:cite:`chevalier_uq`, :cite:`azzimonti_uq`, :cite:`azzimonti_adaptive`.

The main goal is to identify regions in model space where the density field
might be above some given threshold, using the posterior distribution.
We call such regions **excursion set** above the threshold.
We also aim at quantifying the uncertainty on the estimated regions.

-----------------------------------------------------------------------------

The inputs to this module are the posterior mean and the posterior variance,
both as vectors, where the i-th element corresponds to cell nr. i in model
space.

------------------------------------------------------------------------------

.. automodule:: volcapy.uq
.. currentmodule:: volcapy.uq

Module Functionalities
~~~~~~~~~~~~~~~~~~~~~~

========================== ============================================================
Excursion Set Methods
=======================================================================================
coverage_fct                    Compute the excursion probability above a given threshold, at a given point
compute_excursion_probs         For each cell, compute its excursion probability above the given threshold
vorobev_quantile_inds           Get cells belonging to the Vorob'ev quantile at a given level, for a given threshold
vorobev_expectation_inds        Get cells belonging to the Vorob'ev expectation
expected_excursion_measure      Expected measure of excursion set above given threshold
vorobev_deviation               Compute Vorob'ev deviaiton of a given set at a given threshold
========================== ============================================================

Set Uncertainty Quantification: Theory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* The main data characterizing an artificial volcano is a **topography**.
  A topography is a collection of contiguous 3-dimensional cells defining a
  discretization of a given domain (the volcano). For simplicity, we here use a
  cone discretized into uniform cubic cells.

* Once a topography has been defined, one has to chose the locations at which
  the measurements of the gravity field will be performed. We do this by
  picking :math:`n_{obs}` locations at random on the surface of the topography.
  Here surface means the *upper* boundary of the cone, i.e. there will be no
  measurements below the cone. Note also that we add a small offset between the
  surface and the measurement location to avoid singularities in the forwarded
  operator.

* Once topography and measurement locations have been defined, the forward
  operator can be computed using the Banerjee formula.


--------------------------------------------

A detailed description of each functionality is provided below

azzimonti
---------

.. automodule:: volcapy.uq.azzimonti
   :members:

.. bibliography:: bibliography.bib
