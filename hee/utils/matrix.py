import numpy as np
from scipy import sparse

def fast_inverse(M):
    """Computes the inverse of a diagonal matrix.
    :param H: the diagonal matrix to find the inverse of.
    :returns: sparse.csc_matrix -- the inverse of the input matrix as a
            sparse matrix.
    """
    diags = M.diagonal()
    new_diag = []
    for value in diags:
        new_diag.append(1.0/value)

    return sparse.diags([new_diag], [0])