# Try running the whole mesh euclidean distance in memory.
import volcapy.dask.my_prof as pf
import volcapy.grid.covariance_tools as cl

from timeit import default_timer as timer


start = timer()
a = cl.compute_mesh_squared_euclidean_distance(pf.coords_x, pf.coords_x,
	pf.coords_y, pf.coords_y, pf.coords_z, pf.coords_z)

end = timer()
print(str((end - start) / 60.0))
F = pf.F

import torch
a = torch.as_tensor(a)
F = torch.as_tensor(F)
F = F.float()

# Matrix multiplication.
c = torch.mm(F, a)

# Elementwise exponential (underscore means in-place).
a = a.exp_()
