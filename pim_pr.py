import numpy as np


def quality(x, y):
    return np.power(np.abs(np.sum(x * np.conjugate(y))) / np.sum(np.abs(x) * np.abs(y)), 2)

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


def internal_pr(b, xc, A, max_iter=50, max_Qint=0.99):
    x_est = xc
    Ap = np.linalg.pinv(A)
    for i in range(max_iter):
        if i > 0:
            y_store = y_est
        y_est = b * np.exp(1j*np.angle(np.dot(A, x_est)))
        x_est = np.dot(Ap, y_est)
        if i > 0 and quality(y_store, y_est) >= max_Qint:
            break
    return x_est


def internal_pr_svd(b, xc, A, max_iter=50, max_Qint=0.99):
    x_est = xc
    y_est = b * np.exp(1j*np.angle(np.dot(A, x_est)))
    U, S, Vh = np.linalg.svd(A, full_matrices=False)

    for i in range(max_iter):
        if i > 0:
            y_store = y_est
        y_est = b * np.exp(1j*np.angle( np.dot( U, np.dot(U.conj().T, y_est) ) ))
        if i > 0 and quality(y_store, y_est) >= max_Qint:
            break
    
    x_est = np.dot(Vh.conj().T, np.dot(U.conj().T, y_est) / S)
    return x_est


def pim_pr(x, xc, A, max_iter=50, max_Q=0.99):
    for _ in range(max_iter):
        b = np.abs(np.dot(A, x))
        x_est = internal_pr(b, xc, A)
        x = np.abs(x) * np.exp(1j * (np.angle(x) - np.angle(x_est) + np.angle(xc)))
        q = quality(x, xc)
        print(f"Quality: {q}")
        if q > max_Q:
            break
    return x


def pim_pr_svd(x, xc, A, max_iter=50, max_Q=0.99):
    for _ in range(max_iter):
        b = np.abs(np.dot(A, x))
        x_est = internal_pr_svd(b, xc, A)
        x = np.abs(x) * np.exp(1j * (np.angle(x) - np.angle(x_est) + np.angle(xc)))
        q = quality(x, xc)
        print(f"Quality: {q}")
        if q > max_Q:
            break
    return x


def pim_retrieval(b, A, max_iter=50, max_Q=0.99):
    Ap = np.linalg.pinv(A)
    x = np.exp(1j * 2 * np.pi * np.random.rand(A.shape[1]))
    x_est = x

    for _ in range(max_iter):
        x_old = x
        y = b * np.exp(1j * np.angle(np.dot(A, x_est)))
        x = np.dot(Ap, y)
        x_est = np.abs(x) * np.exp(1j * (np.angle(x_est) - np.angle(x)))
        if quality(x, x_old >= max_Q):
            break
    return x
