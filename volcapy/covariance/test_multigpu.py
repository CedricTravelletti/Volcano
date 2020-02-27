from volcapy.inverse.inverse_problem import InverseProblem
import volcapy.covariance.matern32 as cl


import os
os.environ["OMP_NUM_THREADS"] = "128"
os.environ["OPENBLAS_NUM_THREADS"] = "128"
os.environ["MKL_NUM_THREADS"] = "128"
os.environ["VECLIB_NUM_THREADS"] = "128"
os.environ["NUMEXPR_NUM_THREADS"] = "128"


import numpy as np


def main():
    # niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
    niklas_data_path = "/home/ubuntu/Dev/Data/Cedric.mat"
    # niklas_data_path = "/idiap/temp/ctravelletti/tflow/Volcano/data/Cedric.mat"
    inverseProblem = InverseProblem.from_matfile(niklas_data_path)

    coords = inverseProblem.cells_coords[:10000,:]
    lambda0 = 320.0
    n_procs = 128
    cov = cl.compute_cov_cpu(lambda0, coords, n_procs)
    print(cov)
    svd = np.linalg.svd(cov)
    print(svd)


if __name__ == "__main__":
    main()
