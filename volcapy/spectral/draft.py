""" First try at implementing the sparse spectrum Gaussian Process approach
from Lazaro-Gredilla et al (2010).

"""
from volcapy.inverse.flow import InverseProblem
import volcapy.grid.covariance_tools as cl
import numpy as np

# GLOBALS
m0 = 2200.0
data_std = 0.1
sigma_epsilon = data_std


n_freqs = 1000
n_dims = 3

# Initialize an inverse problem from Niklas's data.
# This gives us the forward and the coordinates of the inversion cells.
niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
# niklas_data_path = "/home/ubuntu/Volcano/data/Cedric.mat"
inverseProblem = InverseProblem.from_matfile(niklas_data_path)

# Subset the data.
rest_forward, rest_data = inverseProblem.subset_data(500)

n_model = inverseProblem.n_model

# Now torch in da place.
import torch
torch.set_num_threads(3)

F = torch.as_tensor(inverseProblem.forward)

data_cov = torch.mul(data_std**2, torch.eye(inverseProblem.n_data))

# Careful: we have to make a column vector here.
d_obs = torch.as_tensor(inverseProblem.data_values[:, None])


# Important params.
sigma_0 = 1.0


# Begin computations here.


class SquaredExpModel(torch.nn.Module):
    def __init__(self):
        super(SquaredExpModel, self).__init__()

        self.m0 = torch.nn.Parameter(torch.tensor(m0))
        self.sigma_0 = torch.nn.Parameter(torch.tensor(sigma_0))

        self.m_prior = torch.mul(self.m0, torch.ones((inverseProblem.n_model, 1)))

        self.data_cov = data_cov

        # Initial frequencies.
        # We have to sample them from a multivariate normal.
        # WARNING: The original paper optimizes frequencies and length scale
        # jointly (length scale is the one defining the normal we sample the
        # frequencies from.
        # We do not do it for the moment, and assume the length scale is 200m.
        length_scale = 200.0
        freqs_cov = 1 / (2 * np.pi)**2 * 1 / length_scale**2

        # Define the parameters of the distribution.
        freqs_mean = np.zeros(n_dims)
        freqs_cov = freqs_cov * np.eye(n_dims)
        # Sample
        freqs = np.random.multivariate_normal(freqs_mean, freqs_cov,
                size=n_freqs)
        freqs = freqs.astype(np.float32)
        self.freqs = torch.nn.Parameter(torch.from_numpy(freqs))

        # Put all points in an array.
        # We transpose so the array is n_dim x n_cells.
        coords = torch.as_tensor(inverseProblem.cells_coords).t()

        # Dot product each frequency with each cell. Due to our clever transposition
        # above, this is just a matmul.
        # Resulting matrix is n_freq * n_cells.
        phases = torch.mul(np.float32(np.pi), torch.mm(self.freqs, coords))

        cosines = torch.cos(phases)
        sines = torch.sin(phases)

        # Finally build the phi vector (our principal object of interest).
        # Note that there is a slight difference with the paper: they alternate cos and
        # sin, where we put all the sins after all the coss.
        self.PHI = torch.cat((cosines, sines), dim=0)

    def forward(self):
        """ Squared exponential kernel. Builds the full covariance matrix from a
        squared distance mesh.

        Parameters
        ----------
        lambda_2: float
            Length scale parameter, in the form 1(2 * lambda^2).
        sigma_2: float
            (Square of) standard deviation.

        """
        prior_misfit = torch.sub(d_obs, torch.mm(F, self.m_prior))

        # --------------------------------------------------------------
        fourier_space_fwd = torch.mm(F, self.PHI.t())

        a = torch.mul(
            n_freqs,
            torch.div(sigma_epsilon**2, self.sigma_0**2))

        inv = torch.inverse(
            torch.add(
                torch.mul(a, torch.eye(2 * n_freqs)),
                torch.mm(
                    fourier_space_fwd.t(),
                    torch.mm(
                        torch.eye(inverseProblem.n_data),
                        fourier_space_fwd))))

        tmp = torch.sub(
                torch.eye(inverseProblem.n_data),
                torch.mm(fourier_space_fwd, torch.mm(inv, fourier_space_fwd.t())))

        posterior_mean = torch.add(
            self.m_prior,
            torch.div(
                torch.mm(
                    self.PHI.t(),
                    torch.mm(
                        fourier_space_fwd.t(),
                        torch.mm(
                            tmp,
                            torch.sub(d_obs, torch.mm(F, self.m_prior))))),
                a))

        return posterior_mean


myModel = SquaredExpModel()
# optimizer = torch.optim.SGD(myModel.parameters(), lr = 0.5)
optimizer = torch.optim.Adam(myModel.parameters(), lr=0.3)
criterion = torch.nn.MSELoss()

for epoch in range(600):

    # Forward pass: Compute predicted y by passing
    # x to the model
    posterior_mean = myModel()
    pred_y = torch.mm(F, posterior_mean)

    # Compute and print loss
    loss = criterion(pred_y, d_obs)

    # Loss on test set.
    test_loss = torch.sqrt(criterion(
        torch.mm(
            torch.as_tensor(rest_forward), posterior_mean),
        torch.as_tensor(rest_data)))

    # Zero gradients, perform a backward pass,
    # and update the weights.
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    # loss.backward()
    optimizer.step()
    print(
            'epoch {}, loss {}, m0 {} freq_00 {}, sigma {}'.format(
                    epoch, loss.data,
                    float(myModel.m0),
                    float(myModel.freqs[0, 0]),
                    float(myModel.sigma_0)
                    ))
    print(
            'RMSE on test data:  {}'.format(
                    float(test_loss),
                    ))
