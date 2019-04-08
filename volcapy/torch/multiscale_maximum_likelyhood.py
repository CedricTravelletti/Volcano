# File: run_mesh_distance.py, Author: Cedric Travelletti, Date: 06.03.2019.
""" Run maximum likelyhood parameter estimation. Use concentrated version to avoid
optimizing the prio mean.

"""
from volcapy.inverse.flow import InverseProblem
import volcapy.grid.covariance_tools as cl
from volcapy.grid.regridding import irregular_regrid_single_step, regrid_forward

# Now torch in da place.
import torch
torch.set_num_threads(4)

# GLOBALS
m0 = 2200.0
data_std = 0.1

length_scale = 100.0
sigma = 20.0


# Initialize an inverse problem from Niklas's data.
# This gives us the forward and the coordinates of the inversion cells.
niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
# niklas_data_path = "/home/ubuntu/Volcano/data/Cedric.mat"
inverseProblem = InverseProblem.from_matfile(niklas_data_path)

# Regrid the problem at lower resolution.
coarse_cells_coords, coarse_to_fine_inds = irregular_regrid_single_step(
        inverseProblem.cells_coords, 50.0)

# Train-validation split.
# Save a regridded version before splitting
F_coarse_tot = regrid_forward(inverseProblem.forward, coarse_to_fine_inds)
np.save(F_coarse_tot, "F_coarse_tot.npy")

nr_train = 500
F_test_raw, d_obs_valid_raw = inverseProblem.subset_data(nr_train)
n_data = inverseProblem.n_data

new_F = regrid_forward(inverseProblem.forward, coarse_to_fine_inds)
new_F_test = regrid_forward(F_test_raw, coarse_to_fine_inds)

n_model = len(coarse_to_fine_inds)
del(coarse_to_fine_inds)
F_train_raw = new_F
d_obs_train_raw = inverseProblem.data_values[:, None]

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


class SquaredExpModel(torch.nn.Module):
    def __init__(self):
        super(SquaredExpModel, self).__init__()

        self.m0 = torch.nn.Parameter(torch.tensor(m0))
        self.length_scale = torch.nn.Parameter(torch.tensor(length_scale))
        self.sigma = torch.nn.Parameter(torch.tensor(sigma))

        self.m_prior = torch.mul(self.m0, torch.ones((n_model, 1)))

        self.F = F
        self.d_obs = d_obs
        self.data_cov = data_cov

    def forward(self, distance_mesh):
        """ Squared exponential kernel. Builds the full covariance matrix from a
        squared distance mesh.

        Parameters
        ----------
        lambda_2: float
            Length scale parameter, in the form 1(2 * lambda^2).
        sigma_2: float
            (Square of) standard deviation.

        """
        pushforward_cov = torch.mm(
                torch.mul(
                    torch.exp(
                            torch.mul(distance_mesh,
                                    - 1/(2 * self.length_scale.pow(2)))
                            ),
                self.sigma.pow(2)),
                self.F.t()
                )
        inv_inversion_operator = torch.add(
                        self.data_cov,
                        torch.mm(self.F, pushforward_cov)
                        )
        inversion_operator = torch.inverse(inv_inversion_operator)

        # Need to do it this way, otherwise rounding errors kill everything.
        log_det = - torch.logdet(inversion_operator)

        prior_misfit = torch.sub(self.d_obs, torch.mm(self.F, self.m_prior))

        m_posterior = torch.add(
                self.m_prior,
                torch.mm(
                        torch.mm(pushforward_cov, inversion_operator),
                        prior_misfit)
                )

        # Maximum likelyhood estimator of posterior mean, given values
        # of sigma and lambda. Obtained using the concentration formula.
        log_likelyhood = torch.add(
              log_det,
              torch.mm(
                      prior_misfit.t(),
                      torch.mm(inversion_operator, prior_misfit)))

        return (log_likelyhood, m_posterior)


myModel = SquaredExpModel()
# optimizer = torch.optim.SGD(myModel.parameters(), lr = 0.5)
optimizer = torch.optim.Adam(myModel.parameters(), lr=4.0)

for epoch in range(200):

    # Forward pass: Compute predicted y by passing
    # x to the model
    tmp = myModel(distance_mesh)
    log_likelyhood = tmp[0]
    m_posterior = tmp[1]

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

    print(
            'epoch {}, loss {}, m0 {} length_scale {}, sigma {}'.format(
                    epoch, loss.data,
                    float(myModel.m0),
                    float(myModel.length_scale),
                    float(myModel.sigma)
                    ))
    print("RMSE prediction error: {}".format(prediction_error))

print("Saving Results ...")
torch.save(m_posterior, "posterior_mean.pt")
torch.save(myModel.m0, "m0.pt")
torch.save(myModel.sigma_0, "sigma_0.pt")
torch.save(myModel.length_scale, "length_scale.pt")
