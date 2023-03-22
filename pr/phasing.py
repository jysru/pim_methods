import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import quality


def spgd(x, dither: float = 0.2, max_iter: int = 100, gain: float = 4) -> np.array:
    J = np.zeros(shape=(max_iter,))
    grad = np.zeros_like(J)

    for i in range(max_iter):
        sign_dither = np.sign(np.random.rand(*x.shape) - 0.5)
        x_plus, x_minus = x * np.exp(+1j * sign_dither * dither), x * np.exp(-1j * sign_dither * dither)
        J_plus, J_minus = quality(x_plus, np.abs(x)), quality(x_minus, np.abs(x))
        grad[i] = (J_plus - J_minus) / (2 * dither)
        x = x * np.exp(1j * gain * grad[i] * sign_dither * dither)
        J[i] = quality(x, np.abs(x))

    return J, grad


def kN_tests(x, dither: float = 0.2, loops: int = 3):
    x0 = x.copy()
    acts = x.shape[0]
    max_iter = loops * acts
    J = np.zeros(shape=(max_iter,))

    i = 0
    for _ in range(loops):
        for act in range(acts):
            x_plus, x_minus = x.copy(), x.copy()
            x_plus[act] = np.array(x[act] * np.exp(+1j * dither))
            x_minus[act] = np.array(x[act] * np.exp(-1j * dither))

            J0, J_plus, J_minus = quality(x, np.abs(x0)), quality(x_plus, np.abs(x0)), quality(x_minus, np.abs(x0))
            Re = (2*J0 - J_plus - J_minus) / (4 * (1 - np.cos(dither)))
            Im = (J_plus - J_minus) / (4 * np.sin(dither))
            corr = np.angle(Re + 1j * Im)
            x[act] = x[act] * np.exp(1j * corr)
            J[i] = quality(x, np.abs(x0))
            i += 1

    return J



if __name__ == "__main__":
    n = 10
    n_loops = 3
    x = np.ones(shape=(n,)) * np.exp(1j * 2 * np.pi * np.random.rand(n))
    dither = 0.1

    J1, _ = spgd(x.copy(), dither=dither, max_iter=n*n_loops, gain=5)
    J2 = kN_tests(x.copy(), dither=dither, loops=n_loops)

    plt.figure()
    ax = plt.gca()
    ax.plot(J1, label='SPGD')
    ax.plot(J2, label='kN tests')
    ax.set_xlabel('Actuation #')
    ax.set_ylabel('Metric')
    ax.set_title('Algorithms comparison')
    plt.legend()
    plt.show()