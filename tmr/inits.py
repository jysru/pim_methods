import numpy as np


def random_init(n, m):
    return np.random.rand(m,n) * np.exp(1j * 2 * np.pi * np.random.rand(m,n))

def wirtinger_initialization(A, B):
    n, m, N = A.shape[1], B.shape[1], B.shape[0]
    X_init = np.zeros(shape=(n, m), dtype=np.complex64)
    B = np.square(B)

    for k in range(m):
        lbda = 0
        for j in range(N):
            lbda += np.square(np.linalg.norm(A[j, :]))
        lbda = np.sqrt(n * np.sum(B[:, k]) / lbda)
        _, v = np.linalg.eig(np.dot(np.transpose(np.conj(A)), np.dot(np.diag(B[:, k]), A)) / N)
        X_init[:, k] = lbda * v[:, 0]

    return X_init