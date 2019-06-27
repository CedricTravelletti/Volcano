""" Tools to compute covariance matrix (data side) and covariance pushforward,
on GPU.

"""
import torch

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

    inv_lambda2 = - 1 / (2 * lambda0**2)
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
        if i % n_flush == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Squared euclidean distance.
        d_2 = torch.pow(
            x.unsqueeze(1).expand(x.shape[0], n_model, n_dims)
            - cells_coords.unsqueeze(0).expand(x.shape[0], n_model, n_dims)
            , 2).sum(2)
        tot = torch.cat((
                tot,
                torch.matmul(
                    torch.exp(inv_lambda2 * d_2)
                    , F.t())))

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Send back to CPU.

    end = timer()
    print((end - start)/60.0)

    return tot.cpu()

def compute_K_d(lambda0, F, cells_coords, device, n_chunks=200):
    """ Compute the covariance pushforward.

    The covariance pushforward is just KF^T, where K is the model
    covariance matrix.

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

    inv_lambda2 = - 1 / (2 * lambda0**2)
    n_dims = 3

    # Array to hold the results. We will compute line by line and concatenate.
    tot = torch.Tensor().to(device)

    # Compute K * F^T chunk by chunk.
    # That is, of all the cell couples, we compute the distance between some
    # cells (here x) and ALL other cells. Then repeat for other chunk and
    # concatenate.
    for i, x in enumerate(torch.chunk(cells_coords, chunks=n_chunks, dim=0)):
        # Empty cache every so often. Otherwise we get out of memory errors.
        if i % 50 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Squared euclidean distance.
        d_2 = torch.pow(
            x.unsqueeze(1).expand(x.shape[0], n_model, n_dims)
            - cells_coords.unsqueeze(0).expand(x.shape[0], n_model, n_dims)
            , 2).sum(2)
        tot = torch.cat((
                tot,
                torch.matmul(
                    F,
                    torch.matmul(
                        torch.exp(inv_lambda2 * d_2),
                        F.t()))
                ))

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Send back to CPU.

    end = timer()
    print((end - start)/60.0)

    return tot.cpu()
