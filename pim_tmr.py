import numpy as np
import matplotlib.pyplot as plt

def quality(x, y):
    return np.power(np.abs(np.sum(x * np.conjugate(y))) / np.sum(np.abs(x) * np.abs(y)), 2)

def mse(x, y):
    return np.square(x - y).mean()

def mse_cols(x, y):
    return np.mean(np.square(x - y), axis=0)

def random_init(n, m):
    return np.random.rand(m,n)*np.exp(1j*2*np.pi*np.random.rand(m,n))

def init_wirtinger(A, B):
    pass


def pim_tmr(A, B, max_iter=10000, tol=1e-3, tol_stag=1e-3, max_stag=10):
    n, m = A.shape[1], B.shape[1]
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    S = np.diag(s)
    Bnorm = B/np.linalg.norm(B, ord='fro')

    Xk = random_init(m, n)
    active_cols = np.ones(shape=(m), dtype=bool)
    converged_cols = np.logical_not(active_cols)
    stag_cols = np.zeros(shape=(m), dtype=bool)
    i_stags = np.zeros(shape=(m), dtype=int)
    
    normres = np.zeros(shape=(m), dtype=float)
    restart = 0
    for iter in range(max_iter):
        tmp = B[:, active_cols] * np.exp(1j * np.angle(np.dot(A, Xk[:, active_cols])))
        tmp = np.dot(U.conj().T, tmp) 
        Xk[:, active_cols] = np.dot(Vh.conj().T, np.linalg.solve(S, tmp) )

        Bk = np.abs(np.dot(A, Xk))
        betak = mse_cols(Bk/np.linalg.norm(Bk, ord='fro'), Bnorm)
        MSE = mse(Bk/np.linalg.norm(Bk, ord='fro'), Bnorm)

        convergence_check = (betak < tol)
        converged_cols[active_cols] = convergence_check[active_cols] 
        active_cols = np.logical_not(converged_cols)

        normresanc = normres
        normres = betak

        stag_cols[active_cols] = (np.abs(normres[active_cols] - normresanc[active_cols])/normres[active_cols] < tol_stag)
        i_stags[stag_cols] = i_stags[stag_cols] + 1

        cols_reset = (i_stags >= max_stag)
        if np.any(cols_reset):
            Xnew = random_init(np.sum(cols_reset), n)
            Xk[:, cols_reset] = Xnew
            print(f"Restarting {np.sum(cols_reset)} columns ")
            i_stags[cols_reset] = 0
            restart += 1

        print(f"i={iter:5.0f}  mse_glob={MSE:1.3e}  mse_convs={np.mean(betak[converged_cols]):1.3e}  mse_acts={np.mean(betak[active_cols]):1.3e}  act={np.sum(active_cols):5.0f}  convs={np.sum(converged_cols):5.0f}")

        if np.sum(active_cols) == 0:
            break

    return Xk


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
    diff_angle = np.angle(np.exp(1j*(np.angle(X1.T) - np.angle(X2.T))))
    diff_means = np.mean(diff_angle, axis=1).reshape((diff_angle.shape[0],1))
    diff_angle = np.angle(np.exp(1j*(diff_angle - diff_means)))
    pl1 = axs[1].imshow(diff_angle, cmap="twilight")

    plt.colorbar(pl0, ax=axs[0])
    plt.colorbar(pl1, ax=axs[1])

    plt.show()


if __name__ == "__main__":
    N = 2000
    n = 32
    # m = 288*n
    m = 12*n

    A = np.random.rand(N,n)*np.exp(1j*2*np.pi*np.random.rand(N,n))
    X = np.random.rand(n,m)*np.exp(1j*2*np.pi*np.random.rand(n,m))
    B = np.abs(np.dot(A, X))
    # B = np.abs(B + 0.1*np.random.randn(B.shape[0], B.shape[1]))

    X_est = pim_tmr(A, B, max_iter=1000, tol=8e-9, max_stag=20)
    B_est = np.abs(np.dot(A, X_est))
    print(f"MSE Bs: {mse(B, B_est):0.3e}")

    compare_matrices(X, X_est)
    plt.show()

    