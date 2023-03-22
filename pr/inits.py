import numpy as np


def wirtinger_initialization(b, A):
    m, n = A.shape[0], A.shape[1]
    b = np.square(b)
    
    lbda = 0
    for i in range(m):
        lbda += np.square(np.linalg.norm(A[i, :])) 
    lbda = np.sqrt(n * np.sum(b) / lbda)

    _, v = np.linalg.eig(np.dot(np.transpose(np.conj(A)), np.dot(np.diag(b), A)) / m)
    x_init = lbda * v[:, 0]
    return x_init


def power_spectrum_initialization(b, A, iter=100):
    m, n = A.shape[0], A.shape[1]
    b = np.square(b)

    x_init = np.random.randn(n)
    x_init = x_init / np.linalg.norm(x_init)

    for i in range(iter):
        x_init = np.dot(np.transpose(np.conj(A)), (b * np.dot(A, x_init)))
        x_init = x_init / np.linalg.norm(x_init)

    normest = np.sqrt(np.sum(b)/b.size)
    return x_init * normest


def null_initialization(b, A, iter=100):
    m, n = A.shape[0], A.shape[1]
    b = np.square(b)

    x_init = np.random.randn(n)
    x_init = x_init / np.linalg.norm(x_init)

    for i in range(iter):
        x_init = np.dot(np.transpose(np.conj(A)), (np.ones_like(b) * np.dot(A, x_init)))
        x_init = x_init / np.linalg.norm(x_init)

    normest = np.sqrt(np.sum(b)/b.size)
    return x_init * normest
