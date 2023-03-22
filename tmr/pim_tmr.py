import numpy as np
import matplotlib.pyplot as plt
from utils.utils import compare_matrices
from inits import random_init, wirtinger_initialization
from utils.metrics import quality, mse, mse_cols


def pim_tmr(A, B, max_iter: int = 10000, tol: float = 1e-3, tol_stag: float = 1e-3, max_stag: int = 10, init_matrix = None, init_wirtinger: bool = False, disable_outputs: bool = False, col_per_col: bool = True):
    n, m = A.shape[1], B.shape[1]
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    S = np.diag(s)
    Bnorm = B / np.linalg.norm(B, ord='fro')

    if init_matrix is not None:
        Xk = init_matrix
    else:
        if init_wirtinger:
            Xk = wirtinger_initialization(A, B)
        else:
            Xk = random_init(m, n)
    active_cols = np.ones(shape=(m), dtype=bool)
    converged_cols = np.logical_not(active_cols)
    stag_cols = np.zeros(shape=(m), dtype=bool)
    i_stags = np.zeros(shape=(m), dtype=int)
    
    normres = np.zeros(shape=(m), dtype=float)
    restart = 0
    i_stag = 0
    for iter in range(max_iter):
        tmp = B[:, active_cols] * np.exp(1j * np.angle(np.dot(A, Xk[:, active_cols])))
        tmp = np.dot(U.conj().T, tmp) 
        Xk[:, active_cols] = np.dot(Vh.conj().T, np.linalg.solve(S, tmp) )

        Bk = np.abs(np.dot(A, Xk))
        betak = mse_cols(Bk / np.linalg.norm(Bk, ord='fro'), Bnorm)
        MSE = mse(Bk / np.linalg.norm(Bk, ord='fro'), Bnorm)

        convergence_check = (betak < tol)
        converged_cols[active_cols] = convergence_check[active_cols] 
        active_cols = np.logical_not(converged_cols)

        normresanc = normres
        normres = betak

        if col_per_col:
            stag_cols[active_cols] = (np.abs(normres[active_cols] - normresanc[active_cols]) / normres[active_cols] < tol_stag)
            i_stags[stag_cols] = i_stags[stag_cols] + 1

            cols_reset = (i_stags >= max_stag)
            if np.any(cols_reset):
                Xnew = random_init(np.sum(cols_reset), n)
                Xk[:, cols_reset] = Xnew
                if not disable_outputs:
                    print(f"Restarting {np.sum(cols_reset)} columns ")
                i_stags[cols_reset] = 0
                restart += 1
        else:
            if np.any((np.abs(normres[active_cols] - normresanc[active_cols]) / normres[active_cols] < tol_stag)):
                i_stag += 1
            if i_stag > max_stag:
                Xnew = random_init(np.sum(active_cols), n)
                Xk[:, active_cols] = Xnew
                i_stag = 0
                restart += 1

        if not disable_outputs:
            print(f"i={iter:5.0f}  mse_glob={MSE:1.3e}  mse_convs={np.mean(betak[converged_cols]):1.3e}  mse_acts={np.mean(betak[active_cols]):1.3e}  act={np.sum(active_cols):5.0f}  convs={np.sum(converged_cols):5.0f}")

        if np.sum(active_cols) == 0:
            break

    return Xk





if __name__ == "__main__":
    N = 2000
    n = 32
    m = 12*n

    A = np.random.rand(N,n) * np.exp(1j * 2 * np.pi * np.random.rand(N,n))
    X = np.random.rand(n,m) * np.exp(1j * 2 * np.pi * np.random.rand(n,m))
    B = np.abs(np.dot(A, X))

    X_est = pim_tmr(A, B, max_iter=1000, tol=8e-9, max_stag=20)
    B_est = np.abs(np.dot(A, X_est))
    print(f"MSE Bs: {mse(B, B_est):0.3e}")

    compare_matrices(X, X_est)
    plt.show()

    