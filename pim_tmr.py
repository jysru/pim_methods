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

def pim_tmr(A, B, max_iter=10000, max_Q=0.99, tol=1e-3):
    n, m = A.shape[1], B.shape[1]
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    S = np.diag(s)

    Xk = random_init(m, n)
    J = np.ones(shape=(m), dtype=bool)
    

    normres = 0
    nm = 0
    nmMax = 10
    restart = 0
    for iter in range(max_iter):
        # Xk(:,J) = V*(S\(U'*(pam.B(:,J).*exp(1i*angle(pam.A*Xk(:,J))))));
        tmp = B[:, J] * np.exp(1j * np.angle(np.dot(A, Xk[:, J])))
        tmp = np.dot(U.conj().T, tmp) 
        Xk[:, J] = np.dot(Vh.conj().T, np.linalg.solve(S, tmp) )

        Bk = np.abs(np.dot(A, Xk))
        res = np.abs(Bk - B)
        normresanc = normres
        normres = np.linalg.norm(res, ord=1)
        I = (np.sum(res, axis=0) < tol)
        J = np.logical_not(I)
        j = np.sum(J)

        if (np.abs(normres - normresanc)/normres < 0.01*tol):
            nm += 1
        print(f"{iter:5.0f}  {normres:10.4e} {j:5.0f}")

        if nm >= nmMax:
            Xnew = random_init(m, n)
            Xk[:, J] = Xnew[:, J]
            print(f"Restart")
            nm = 0
            restart += 1

        if j == 0:
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
    N = 500
    n = 32
    m = 288*n

    A = np.random.rand(N,n)*np.exp(1j*2*np.pi*np.random.rand(N,n))
    X = np.random.rand(n,m)*np.exp(1j*2*np.pi*np.random.rand(n,m))
    B = np.abs(np.dot(A, X))

    X_est = pim_tmr(A, B)
    B_est = np.abs(np.dot(A, X_est))
    print(f"MSE Bs: {mse(B, B_est):0.3e}")

    # compare_matrices(X, X_est)
    # plt.show()

    