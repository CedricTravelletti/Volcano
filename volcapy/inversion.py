""" Tools to perform Bayesian inversion (gaussian linear settin)
in model or data space.
"""
import numpy as np


def inversion_model(m_prior, cov_m, d_obs, cov_d, forward_op):
    """ Perform inversion in model space.
    """
    return (m_prior
            + cov_m @ forward_op.T @ np.linalg.inv(
                    forward_op @ cov_m @ forward_op.T + cov_d)
            @ (d_obs - forward_op @ m_prior))

def inversion_data(m_prior, cov_m, d_obs, cov_d, forward_op):
    """ Perform inversion in data space.
    """
    return (m_prior
            + np.linalg.inv(forward_op.T @ np.linalg.inv(cov_d) @ forward_op
                    + np.linalg.inv(cov_m)) @ forward_op.T @ np.linalg.inv(cov_d) @ (d_obs - forward_op @ m_prior))
