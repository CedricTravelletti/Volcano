""" Tools to compute covariance matrix (data side) and covariance pushforward,
on GPU.

IMPORTANT: Note that we always strip the variance parameter sigma0 from the
covariance matrix. Hence, when using the covariance pushforward computed here,
one has to manually multiply by :math:`\sigma_0^2:: for expressions to make sense.

"""
