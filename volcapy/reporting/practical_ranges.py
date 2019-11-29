""" Compute practical range of kernel.

"""
import volcapy.covariance.squared_exponential as cov
import numpy as np
from scipy import optimize
import torch


lambda0 = 342.0

# Function to optimize to find practical range.
def f(x):
    coords = torch.from_numpy(np.array([[0,0,0], [0,0, x]]))
    return cov.compute_cov(lambda0, coords, 0, 1) - 0.5

if __name__ == "__main__":
    sol = optimize.root_scalar(f, bracket=[0, 10000], method='brentq')
    print("Root {}".format(sol.root))

    # Check.
    coords = torch.from_numpy(np.array([[0,0,0], [0,0, sol.root]]))
    print(cov.compute_cov(lambda0, coords, 0, 1))
