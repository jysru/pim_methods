import numpy as np


def get_subsets(A, B, N: int, randomize: bool = False):
    dset_length = A.shape[0]

    if randomize:
        idx = np.random.permutation(dset_length)
        return (A[idx[:N], :], B[idx[:N], :])
    else:
        return (A[:N, :], B[:N, :])

