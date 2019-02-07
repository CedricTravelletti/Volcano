# File: matrix_tools.py, Author: Cedric Travelletti, Date: 30.01.2019.
""" Tools to work with BIG matrices or implicitly defined ones.

An *implicitly defined matrix* is a matrix A defined by a function: f(i, j) =
A_ij. This allows one to work with matrices that are too big to fit in memory
by only creating the element i,j when it is needed.

The multiplication of an implicitly defined matrix A with a regular one C can be
efficiently computed by chunking: We use the function f to create the first n
rows of A and multiply them with B. This will give the first n row of the
product. We then proceed to next n rows, etc... .

Implicit Matrices are implemented using the class ImplicitMatrix.
Such an object should provide two methods:
    get_element(i, j) which returns element i, j of the matrix.

    get_rows(start, end) which returns the sub-matrix containing
    rows start to end (included).

Implicit matrix multiplication only makes sense if both those functions are
blazingly fatst. Hence, the way to go is to implement them in C, and then bind
them to the class for nice encapsulation.

"""
from math import floor
import numpy as np


class ImplicitMatrix():
    """ Matrix whose elements are defined by a function.

    Parameters
    ----------
    get_element: function(int, int)
        Functions that returns element i,j: f(i,j) = A_ij.
    get_rows: function(int, int)
        Function returning rows start to end (included) as a numpy matrix.
    shape: (int, int)
        Tuple defining the shape of the matrix.

    """
    def __init__(self, get_element, get_rows, shape):
        # Dynamically bind the functions.
        self.get_element = get_element
        self.get_rows = get_rows

        self.shape = shape

def left_implicit_mat_mult(A_implicit, B, chunk_size):
    """ Computes the matrix product A * B, where A is implicitly defined
    through f.

    Parameters
    ----------
    A: ImplicitMatrix
        An implicitly defined matrix A.
    B: 2D-ndarray
        The matrix on the left of the multiplication.

    Returns
    -------
    2D-ndarray
        The matrix A*B.

    """
    # Create output.
    out = np.zeros((A_implicit.shape[0], B.shape[1]))

    # Create list of chunks.
    n_lines = A_implicit.shape[0]
    chunks = chunk_range(n_lines, chunk_size)

    # For each chunk: Perform partial multiplication and populate output matrix.
    for chunk in chunks:
        # Build some rows from A.
        A_partial = A_implicit.get_rows(chunk[0], chunk[1])

        out[chunk[0]: chunk[1] + 1, :] = A_partial @ B

    return out

# TODO: Would be elegant to improve to a full-fledged iterator.
def chunk_range(n_lines, chunk_size):
    """ Given a range like [1,...,n], chunk it into chunks of a given size.
    The goal of this function is to allow chunked iteration of a range.

    The original use of this function was to iterate lines of a matrix by
    chunks.
    In that context, a chunk is a tuple, whose first element gives the number
    of the first line we wonsider and the last number give the number of the
    last line we consider. Henche the chunk (0,2) means we consider the first 3
    lines of a matrix.

    Parameters
    ----------
    n_lines: int
        The range to chunk. Will chunk the range

    """

    # Create the list of chunks.
    chunks = []

    # We loop from the first to the penultimate chunk.
    for i in range(floor(n_lines / chunk_size)):
        chunks.append((i * chunk_size, i * chunk_size + chunk_size - 1))

    # Last chunk cannot be fully loop.
    chunks.append(
            (
                    floor(n_lines / float(chunk_size))*chunk_size, n_lines)
            )
    return chunks
