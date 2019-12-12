""" The goal of this script is to compute the residual uncertainty left after
filling space with measurements.

i) Use real data to get posterior mean.
ii) Generate fake measurement sites on the surface.
iii) Compute associated forward.
iv) Use posterior mean to generate fake data.
v) Invert the data to get the "space-filling posterior".
vi) Apply the UQ techniques on the space-filling posterior and have a look at
the (residual) Vorob'ev deviation.
