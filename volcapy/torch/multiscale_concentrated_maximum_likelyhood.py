# File: run_mesh_distance.py, Author: Cedric Travelletti, Date: 06.03.2019.
""" Run maximum likelyhood parameter estimation. Use concentrated version to avoid
optimizing the prio mean.

"""
from volcapy.inverse.flow import InverseProblem
import volcapy.grid.covariance_tools as cl
from volcapy.grid.regridding import irregular_regrid_single_step, regrid_forward
import numpy as np

# Set up logging.
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Now torch in da place.
import torch
# Choose between CPU and GPU.
device = torch.device('cuda:0')

# GLOBALS
data_std = 0.1

length_scale = 164.17
sigma_0 = 88.95


# Initialize an inverse problem from Niklas's data.
# This gives us the forward and the coordinates of the inversion cells.
# niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
# niklas_data_path = "/home/ubuntu/Volcano/data/Cedric.mat"
niklas_data_path = "/idiap/temp/ctravelletti/tflow/Volcano/data/Cedric.mat"
inverseProblem = InverseProblem.from_matfile(niklas_data_path)

# Regrid the problem at lower resolution.
coarse_cells_coords, coarse_to_fine_inds = irregular_regrid_single_step(
        inverseProblem.cells_coords, 50.0)

# Save a regridded version before splitting
F_coarse_tot = regrid_forward(inverseProblem.forward, coarse_to_fine_inds)
np.save("F_coarse_tot.npy", F_coarse_tot)

# Train-validation split.
nr_train = 500
F_test_raw, d_obs_valid_raw = inverseProblem.subset_data(nr_train)
d_obs_train_raw = inverseProblem.data_values[:, None]
n_data = inverseProblem.n_data

print("Train/Test split: {} / {}.".format(
        d_obs_train_raw.shape[0], d_obs_valid_raw.shape[0]))
print(inverseProblem.data_values.shape)

new_F = regrid_forward(inverseProblem.forward, coarse_to_fine_inds)
new_F_test = regrid_forward(F_test_raw, coarse_to_fine_inds)

n_model = len(coarse_to_fine_inds)
del(coarse_to_fine_inds)

print("Size of model after regridding: {} cells.".format(n_model))

# Careful: we have to make a column vector here.
d_obs = torch.as_tensor(d_obs_train_raw)
d_obs_test = torch.as_tensor(d_obs_valid_raw)

distance_mesh = cl.compute_mesh_squared_euclidean_distance(
        coarse_cells_coords[:, 0], coarse_cells_coords[:, 0],
        coarse_cells_coords[:, 1], coarse_cells_coords[:, 1],
        coarse_cells_coords[:, 2], coarse_cells_coords[:, 2])

del(coarse_cells_coords)
del(inverseProblem)

distance_mesh = torch.as_tensor(distance_mesh)
F = torch.as_tensor(new_F)
F_test = torch.as_tensor(new_F_test)

data_cov = torch.mul(data_std**2, torch.eye(n_data))

# Distance mesh is horribly expansive, to use half-precision.
distance_mesh = distance_mesh.to(torch.device("cpu"))

class SquaredExpModel(torch.nn.Module):
    def __init__(self):
        super(SquaredExpModel, self).__init__()

        # self.length_scale = torch.nn.Parameter(torch.tensor(length_scale))
        self.length_scale = torch.nn.Parameter(torch.tensor(length_scale))
        self.sigma_0 = torch.nn.Parameter(torch.tensor(sigma_0))

        self.F = F.to(device)
        self.d_obs = d_obs.to(device)
        self.data_cov = data_cov.to(device)

        self.I = torch.ones((n_model, 1), dtype=torch.float32,
                        device=device)


    def forward(self, distance_mesh):
        torch.cuda.empty_cache()
        logger.debug("GPU used before forward pass: {} Gb.".format(
                torch.cuda.memory_allocated(device)/1e9))

        # Distance mesh is horribly expansive, so perform on CPU.
        fact = (- 1/(2 * self.length_scale.pow(2))).to(torch.device("cpu"))
        tmp = torch.mul(distance_mesh, fact)
        tmp = tmp.to(device)
        torch.cuda.empty_cache()

        pushforward_cov = torch.mm(
                torch.exp(tmp),
                self.F.t())
        torch.cuda.empty_cache()

        logger.debug("GPU used after computing pushforward_cov: {} Gb.".format(
                torch.cuda.memory_allocated(device)/1e9))

        inv_inversion_operator = torch.add(
                        self.data_cov,
                        torch.mul(
                            self.sigma_0.pow(2),
                            torch.mm(self.F, pushforward_cov))
                        )
        torch.cuda.empty_cache()

        logger.debug("GPU used after computing inv_inversion_operator: {} Gb.".format(
                torch.cuda.memory_allocated(device)/1e9))

        # Back to full precision.
        inversion_operator = torch.inverse(inv_inversion_operator)
        torch.cuda.empty_cache()

        logger.debug("GPU used after computing inversion_operator: {} Gb.".format(
                torch.cuda.memory_allocated(device)/1e9))

        del inv_inversion_operator
        temp_to_move_outside = torch.mm(pushforward_cov,
                inversion_operator)
        del pushforward_cov

        # Need to do it this way, otherwise rounding errors kill everything.
        log_det = - torch.logdet(inversion_operator)

        # We now concentrate the log-likelihood around m0.
        # Since the concetrated version of the prior mean is big, and since it
        # only plays a role when multiplied wiht the forward, we do not compute
        # it directly.
        concentrated_m0 = torch.mm(
            torch.inverse(
                torch.mm(
                    torch.mm(self.I.t(), self.F.t()),
                    torch.mm(
                        inversion_operator,
                        torch.mm(self.F, self.I)))),
            torch.mm(
                torch.mm(self.d_obs.t(), inversion_operator),
                torch.mm(self.F, self.I)))
        concentrated_m_prior = torch.mul(concentrated_m0, self.I)

        concentrated_prior_misfit = torch.sub(self.d_obs,
                torch.mm(self.F, concentrated_m_prior))

        concentrated_log_likelyhood = torch.add(
              log_det,
              torch.mm(
                      concentrated_prior_misfit.t(),
                      torch.mm(inversion_operator, concentrated_prior_misfit)))

        # Maybe move out of model.
        m_posterior = torch.add(
                concentrated_m_prior,
                torch.mm(
                        temp_to_move_outside,
                        concentrated_prior_misfit)
                )
        torch.cuda.empty_cache()
        logger.debug("GPU at end of forward pass: {} Gb.".format(
                torch.cuda.memory_allocated(device)/1e9))

        return (concentrated_log_likelyhood, concentrated_m0, m_posterior)


myModel = SquaredExpModel()
myModel.cuda()
# optimizer = torch.optim.SGD(myModel.parameters(), lr = 0.5)
optimizer = torch.optim.Adam(myModel.parameters(), lr=2.0)

for epoch in range(100000):

    # Forward pass: Compute predicted y by passing
    # x to the model
    tmp = myModel(distance_mesh)
    log_likelyhood = tmp[0]
    concentrated_m0 = tmp[1]
    m_posterior = tmp[2].to(torch.device("cpu"))

    # Compute and print loss
    loss = log_likelyhood

    # Zero gradients, perform a backward pass,
    # and update the weights.
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    # loss.backward()
    optimizer.step()

    # Compute prediction error.
    criterion = torch.nn.MSELoss()
    prediction_error = torch.sqrt(criterion(
            torch.mm(F_test, m_posterior), d_obs_test))

    train_error = torch.sqrt(criterion(
            torch.mm(F.to(torch.device("cpu")), m_posterior), d_obs))
    print("RMSE train error: {}".format(train_error))

    print(
            'epoch {}, loss {}, m0 {} length_scale {}, sigma {}'.format(
                    epoch, loss.data,
                    float(concentrated_m0),
                    float(myModel.length_scale),
                    float(myModel.sigma_0)
                    ))
    print("RMSE prediction error: {}".format(prediction_error))

print("Saving Results ...")
torch.save(m_posterior, "posterior_mean.pt")
torch.save(concentrated_m0, "concentrated_m0.pt")
torch.save(myModel.sigma_0, "sigma_0.pt")
torch.save(myModel.length_scale, "length_scale.pt")
