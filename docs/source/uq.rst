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

We want to estimate regions in model the space :math:`X` where the matter density field
:math:`Z` is above a given threshold :math:`u_0`.

The posterior distribution of the conditional field gives rise to a random closed set (RACS) :math:`\Gamma`

.. math::

   \Gamma = \lbrace x \in X: \tilde{Z}_x \geq u_0 \rbrace

We can then consider the pointwise probability to belong to the excursion set

* *Coverage Function*

.. math::

   p_{\Gamma}: X \rightarrow [0, 1]

.. math::
  
   p_{\Gamma}(x) := \mathbb{P}[x \in \Gamma]

All our set estimators will be defined using the coverage function.

* *Vorob'ev quantile* at level :math:`\alpha`

.. math::

   Q_{\alpha} := \lbrace x \in X : p_{\Gamma} \geq \alpha \rbrace

--------------------------------------------

Module implementation Details
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: volcapy.uq.azzimonti
   :members:

.. bibliography:: bibliography.bib
