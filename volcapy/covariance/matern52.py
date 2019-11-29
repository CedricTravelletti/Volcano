import torch
import numpy as np

# General torch settings and devices.
torch.set_num_threads(8)
gpu = torch.device('cuda:0')
cpu = torch.device('cpu')

from timeit import default_timer as timer


def compute_cov_pushforward(lambda0, F, cells_coords, device, n_chunks=200,
        n_flush=50):
    """ Compute the covariance pushforward.

    The covariance pushforward is just KF^T, where K is the model
    covariance matrix.

    Note that the sigam0^2 is not included, and one has to manually add it when
    using the covariance pushforward computed here.

    Parameters
    ----------
    lambda0: float
        Lenght-scale parameter
    F: tensor
        Forward operator matrix
    cells_coords: tensor
        n_cells * n_dims: cells coordinates
    device: toch.Device
        Device to perform the computation on, CPU or GPU.
    n_chunks: int
        Number of chunks to split the matrix into.
        Default is 200. Increase if get OOM errors.
    n_flush: int
        Synchronize threads and flush GPU cache every *n_flush* iterations.
        This is necessary to avoid OOM errors.
        Default is 50.

    Returns
    -------
    Tensor
        n_model * n_data covariance pushforward K F^t.
    """
    start = timer()


    # Transfer everything to device.
    lambda0 = torch.tensor(lambda0, requires_grad=False).to(device)
    F = F.to(device)
    cells_coords = cells_coords.to(device)

    inv_lambda2 = - np.sqrt(5) / lambda0
    n_dims = 3
    n_model = F.shape[1]

    # Array to hold the results. We will compute line by line and concatenate.
    tot = torch.Tensor().to(device)

    # Compute K * F^T chunk by chunk.
    # That is, of all the cell couples, we compute the distance between some
    # cells (here x) and ALL other cells. Then repeat for other chunk and
    # concatenate.
    for i, x in enumerate(torch.chunk(cells_coords, chunks=n_chunks, dim=0)):
        # Empty cache every so often. Otherwise we get out of memory errors.
        if i % n_flush == 0 and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Euclidean distance.
        d = torch.sqrt(torch.pow(
            x.unsqueeze(1).expand(x.shape[0], n_model, n_dims)
            - cells_coords.unsqueeze(0).expand(x.shape[0], n_model, n_dims)
            , 2).sum(2))
        tot = torch.cat((
                tot,
                torch.matmul(
                    torch.mul(
                        (
                            torch.ones(d.shape, device=device)
                            - inv_lambda2 * d
                            + (1/3) * inv_lambda2**2 * d**2),
                        torch.exp(inv_lambda2 * d))
                    , F.t())))

    # Wait for all threads to complete.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    end = timer()
    print((end - start)/60.0)

    return tot.cpu()

def compute_cov(lambda0, cells_coords, i, j):
    """ Compute the covariance between two points.

    Note that, as always, sigma0 has been stripped.

    Parameters
    ----------
    lambda0: float
        Lenght-scale parameter
    cells_coords: tensor
        n_cells * n_dims: cells coordinates
    i: int
        Index of first cell (index in the cells_coords array).
    j: int
        Index of second cell.

    Returns
    -------
    Tensor
        (Stripped) covariance between cell nr i and cell nr j.
    """
    # Convert to torch.
    lambda0 = torch.tensor(lambda0, requires_grad=False)
    inv_lambda2 = - np.sqrt(5) / lambda0

    # Euclidean distance.
    d = torch.sqrt(torch.pow(
            cells_coords[i, :] - cells_coords[j, :]
            , 2).sum())

    return (1 - inv_lambda2 * d + (1/3) * inv_lambda2**2 * d**2) * torch.exp(inv_lambda2 * d)
