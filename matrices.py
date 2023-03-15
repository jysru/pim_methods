import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def random(n, m, complex: bool = False, sparsity: float = 0):
    X = np.random.rand(n, m)
    if complex:
        X = complexify(X)
    if sparsity:
        X = sparsify(X, sparsity)
    return X


def complexify(x):
    return x * np.exp(1j * 2 * np.pi * np.random.rand(*x.shape))


def sparsify(x, sparsity):
    shape = x.shape
    idx = np.random.permutation(np.arange(x.size))
    idx = idx[0:int(x.size*sparsity)]
    x = x.flatten()
    x[idx] = 0
    return np.reshape(x, shape)


def random_toeplitz(n, m, complex: bool = False, sparsity: float = 0, toeplitz_phase: bool = False):
    r, l = random(n, 1), random(m, 1)
    X = linalg.toeplitz(r, l)
    if complex:
        if toeplitz_phase:
            X = X * np.exp(1j * 2 * np.pi * linalg.toeplitz(random(n, 1), random(m, 1)))
        else:
            X = complexify(X)
    if sparsity:
        X = sparsify(X, sparsity)
    return X


def diag_random(n, complex: bool = False):
    X = np.diag(np.random.rand(n))
    if complex:
        X = complexify(X)
    return X


def normalize(X):
    return np.abs(X) * np.exp(1j * (np.angle(X) - np.angle(X[0,:])))


def plot_matrix(X):
    fig, axs = plt.subplots(1, 2)
    pl0 = axs[0].imshow(np.abs(X.T))
    pl1 = axs[1].imshow(np.angle(X.T), cmap="twilight")
    plt.colorbar(pl0, ax=axs[0])
    plt.colorbar(pl1, ax=axs[1])


def compare_matrices(X1, X2):
    fig, axs = plt.subplots(1, 4)
    pl0 = axs[0].imshow(np.abs(X1.T))
    pl1 = axs[1].imshow(np.abs(X2.T))
    pl2 = axs[2].imshow(np.angle(X1.T), cmap="twilight")
    pl3 = axs[3].imshow(np.angle(X2.T), cmap="twilight")
    plt.colorbar(pl0, ax=axs[0])
    plt.colorbar(pl1, ax=axs[1])
    plt.colorbar(pl2, ax=axs[2])
    plt.colorbar(pl3, ax=axs[3])

    fig, axs = plt.subplots(1, 2)
    pl0 = axs[0].imshow(np.abs(X1.T) - np.abs(X2.T))
    diff_angle = np.angle(np.exp(1j * (np.angle(X1.T) - np.angle(X2.T))))
    diff_means = np.mean(diff_angle, axis=1).reshape((diff_angle.shape[0], 1))
    diff_angle = np.angle(np.exp(1j * (diff_angle - diff_means)))
    pl1 = axs[1].imshow(diff_angle, cmap="twilight")

    plt.colorbar(pl0, ax=axs[0])
    plt.colorbar(pl1, ax=axs[1])

    plt.show()