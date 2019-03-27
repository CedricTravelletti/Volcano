# File: test.py, Author: Cedric Travelletti, Date: 26.02.2019.
""" Implements an idea of David to make the rows of F^T sparse: pick a
threshold. Then randomly zero out the elements below the threshold using the
following procedure: Put them to zero with prob p and divide them by p with
probability (1 - p).

Then multiply the row by one of the row or column of Cm and look at the average
value.

"""
from volcapy import loading
from volcapy.inverse.flow import InverseProblem


path_niklas_data = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
inverse_problem = InverseProblem.from_matfile(path_niklas_data)


p = 0.1
threshold = 0.001

# np.abs(F) > threshold

def zero_out(line, p, threshold=1e-6):
    line_copy = np.copy(line)

    for i, x in enumerate(line_copy):
        if np.abs(x) < threshold:
            rnd = np.random.uniform()
            if rnd < p:
                line_copy[i] = 0
            else:
                line_copy[i] = line_copy[i] / (1 - p)
    return line_copy

def run(line, p, threshold, row_nr, n_iter=100):
    """
    Parameters
    ----------
    line: array
        Line from the forward on which to run.
    p: float
        Probability with which we zero out the elements.
    threshold: float
        Only consider the elements with absolute value below threshold, leave
        the others as they are.
    row_nr: int
        Which row of the covariance matrix to pick for the scalar product.

    """
    print("Size of the line: " + str(line.shape)
        + ", number of elements above threshold: "
        + str(np.count_nonzero(np.abs(line) > threshold)))

    cov_line = inverse_problem.build_partial_covariance(row_nr, row_nr)[0, :]
    results = []

    # Put the original result.
    results.append(np.dot(line, cov_line))

    for i in range(n_iter):
        modified_F = zero_out(line, p, threshold)
        print("Number nonzero: " + str(np.count_nonzero(modified_F)))
        results.append(np.dot(modified_F, cov_line))

    return results
