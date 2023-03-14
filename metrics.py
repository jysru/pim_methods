import numpy as np
import matrices as pim_mats


def quality(x, y):
    return np.power(np.abs(np.sum(x * np.conjugate(y))) / np.sum(np.abs(x) * np.abs(y)), 2)

def pearson(x, y):
        x = np.abs(x)
        y = np.abs(y)
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        std_x = np.std(x)
        std_y = np.std(y)
        n = x.size

        s = np.sum((x - mean_x) * (y - mean_y) / n)
        r = s / (std_x * std_y)
        return r


def quality_statistics(X, X_est, N: int = 100, print_result: bool = True):
    n, m = X.shape[0], X.shape[1]
    A = pim_mats.complex_random(N, X.shape[0])
    Y, Y_est = np.dot(A, X), np.dot(A, X_est)

    q = np.zeros(N)
    for i in range(N):
        q[i] = quality(Y[i,:], Y_est[i,:])

    if print_result:
        print(f"Quality: average = {np.mean(q)*100:.2}%, std = {np.std(q)*100:.2}%")
    return np.mean(q), np.std(q)

    
def pearson_statistics(X, X_est, N: int = 100, print_result: bool = True):
    n, m = X.shape[0], X.shape[1]
    A = pim_mats.complex_random(N, X.shape[0])
    Y, Y_est = np.dot(A, X), np.dot(A, X_est)

    q = np.zeros(N)
    for i in range(N):
        q[i] = pearson(Y[i,:], Y_est[i,:])

    if print_result:
        print(f"Pearson: average = {np.mean(q)*100:.5f}%, std = {np.std(q)*100:.5f}%")
    return np.mean(q), np.std(q)