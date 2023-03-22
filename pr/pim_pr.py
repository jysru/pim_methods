import numpy as np
from utils.metrics import quality



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


def pim_retrieval_svd(x, xc, A, max_iter=50, max_Q=0.9999):
    x_ret = xc.copy()
    for _ in range(max_iter):
        b = np.abs(np.dot(A, x))
        x_est = internal_pr_svd(b, xc, A)
        x = np.abs(x) * np.exp(1j * (np.angle(x) - np.angle(x_est) + np.angle(xc)))
        x_ret = np.abs(x_ret) * np.exp(1j * (np.angle(x_ret) + np.angle(x_est)))
        q = quality(x, xc)
        print(f"Quality: {q}")
        if q > max_Q:
            break
    return x_ret
