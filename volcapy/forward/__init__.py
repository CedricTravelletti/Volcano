"""
Forward Modelling
================================
This module takes care of building the forward operator.
It is also responsible for building the discretization on which we will perform
the inversion.

========================== ============================================================
Excursion Set Methods
=======================================================================================
coverage_fct                    Excursion probability of given cell
vorobev_quantile_inds           Cells in Vorobe'v quantile
vorobev_expectation_inds        Cells in Vorobe'v expectation
expected_excursion_measure      Expected measure of excursion set above given threshold
========================== ============================================================

"""
