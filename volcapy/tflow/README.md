## sparse_mesh: Ignore small entries in the covariance matrix.
This is done by ignoring cell couples for which the distance between the cells
is bigger than some threshold.
Practically, for each cell, we query all neighbors within radius.
This is done using a BallTree, which turns out to be extremely fast.

Mathematically, we view the process described above a an approximation of the
quadratic matrix inversion problem. That is, we approximate the kriggin
equations themselves.

The above justification is necessary, since by deleting entries in the kernel
matrix, we lose positive definiteness, hence, technically, we are not in the
krigging world anymore.

## mse_train_chunked.py: Minimize MSE. Model fits in memory thanks to chunking
and applying operations per chunk.

## Chunk vs Line: find optimal chunk size.
