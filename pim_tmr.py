import numpy as np
import matplotlib.pyplot as plt

def quality(x, y):
    return np.power(np.abs(np.sum(x * np.conjugate(y))) / np.sum(np.abs(x) * np.abs(y)), 2)

def mse(x, y):
    return np.square(x - y).mean()

def random_init(n, m):
    return np.random.rand(m,n)*np.exp(1j*2*np.pi*np.random.rand(m,n))

def init_wirtinger(A, B):
    pass

def pim_tmr(A, B, max_iter=10000, tol=1e-3, tol_stag=1e-3, max_stag=10):
    n, m = A.shape[1], B.shape[1]
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    S = np.diag(s)

    Xk = random_init(m, n)
    cols = np.ones(shape=(m), dtype=bool)
    cols_ok = np.zeros(shape=(m), dtype=bool)
    cols_stag = np.ones(shape=(m), dtype=bool)
    i_stags = np.zeros(shape=(m), dtype=int)
    
    normres = np.zeros(shape=(m), dtype=float)
    restart = 0
    for iter in range(max_iter):
        tmp = B[:, cols] * np.exp(1j * np.angle(np.dot(A, Xk[:, cols])))
        tmp = np.dot(U.conj().T, tmp) 
        Xk[:, cols] = np.dot(Vh.conj().T, np.linalg.solve(S, tmp) )

        Bk = np.abs(np.dot(A, Xk))
        Rk = np.abs(Bk/np.linalg.norm(Bk, ord='fro') - B/np.linalg.norm(B, ord='fro'))
        MSE = mse(Bk, B)
        betak = np.sum(Rk, axis=0)

        cols_ok = (betak < tol)
        cols = np.logical_not(cols_ok)
        cols_not_ok = np.sum(cols)

        normresanc = normres
        normres = betak

        cols_stag[cols] = (np.abs(normres[cols] - normresanc[cols])/normres[cols] < tol_stag)
        i_stags[cols_stag] = i_stags[cols_stag] + 1

        cols_reset = (i_stags >= max_stag)
        if np.any(cols_reset):
            Xnew = random_init(np.sum(cols_reset), n)
            Xk[:, cols_reset] = Xnew
            print(f"Restarting {np.sum(cols_reset)} columns ")
            i_stags[cols_reset] = 0
            restart += 1

        print(f"{iter:5.0f} {MSE:10.4e} {np.sum(Rk):10.4e} {cols_not_ok:5.0f}")

        if cols_not_ok == 0:
            break

    return Xk


def compare_matrices(X1, X2):
    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(np.abs(X1.T))
    axs[1].imshow(np.abs(X2.T))
    axs[2].imshow(np.angle(X1.T), cmap="twilight")
    axs[3].imshow(np.angle(X2.T), cmap="twilight")

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(np.abs(X1.T) - np.abs(X2.T))
    diff_angle = np.angle(np.exp(1j*(np.angle(X1.T) - np.angle(X2.T))))
    diff_means = np.mean(diff_angle, axis=1).reshape((diff_angle.shape[0],1))
    diff_angle = np.angle(np.exp(1j*(diff_angle - diff_means)))
    axs[1].imshow(diff_angle, cmap="twilight")

    plt.show()


if __name__ == "__main__":
    N = 200
    n = 32
    # m = 288*n
    m = 8*n

    A = np.random.rand(N,n)*np.exp(1j*2*np.pi*np.random.rand(N,n))
    X = np.random.rand(n,m)*np.exp(1j*2*np.pi*np.random.rand(n,m))
    B = np.abs(np.dot(A, X))

    X_est = pim_tmr(A, B)
    B_est = np.abs(np.dot(A, X_est))
    print(f"MSE Bs: {mse(B, B_est):0.3e}")

    # compare_matrices(X, X_est)
    # plt.show()

    