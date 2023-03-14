import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

def random(n, m):
    return np.random.rand(n, m)


def complex_random(n, m):
    return np.random.rand(n,m) * np.exp(1j * 2 * np.pi * np.random.rand(n,m))


def diag_random(n):
    return np.diag(np.random.rand(n))


def diag_complex_random(n):
    return np.diag(np.random.rand(n))*np.exp(1j * 2 * np.pi * np.random.rand(n))


def sparsify(x, sparsity):
    shape = x.shape
    idx = np.random.permutation(np.arange(x.size))
    idx = idx[0:int(x.size*sparsity)]
    x = x.flatten()
    x[idx] = 0
    return np.reshape(x, shape)


def sparse_random(n, m, sparsity):
    x = random(n, m)
    return sparsify(x, sparsity)


def sparse_complex_random(n, m, sparsity):
    x = complex_random(n, m)
    return sparsify(x, sparsity)


def random_toeplitz(n, m):
    r = random(n, 1)
    l = random(m, 1)
    x = linalg.toeplitz(r, l)
    return x


def random_complex_toeplitz(n, m):
    x = random_toeplitz(n, m)
    phi = np.angle(complex_random(n, m))
    return x * np.exp(1j * phi)


def random_complex_full_toeplitz(n, m):
    x = random_toeplitz(n, m)
    phi = 2 * np.pi * random_toeplitz(n, m)
    return x * np.exp(1j * phi)


def sparse_random_toeplitz(n, m, sparsity):
    x = random_toeplitz(n, m)
    return sparsify(x, sparsity)


def sparse_random_complex_toeplitz(n, m, sparsity):
    x = random_complex_toeplitz(n, m)
    return sparsify(x, sparsity)


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