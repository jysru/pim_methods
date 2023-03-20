import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Adjusted from https://gitlab.xlim.fr/shpakovych/phrt-opt


class TMR():

    def __init__(self, A, B, X_init = None, dtype: str = 'complex64', device: str = 'cpu', verbose: bool = True, save_best: bool = True) -> None:
        self.A = A
        self.B = B
        self.X_init = X_init
        self.dtype = dtype
        self.X = None
        self.X_best = None
        self.verbose = verbose
        self.device = device
        self.save_best = save_best
        self.metric = []
        self.iter = []
        self.mse = []

    def _check_cuda(self) -> bool:
        return torch.cuda.is_available()
    
    def run_admm(self,
                 rho=0.25,
                 max_iter=100,
                 max_inner_iter=100,
                 tol=1e-11,
                 lr=1e-2,
                 dtype='complex64',
                 is_lmd_cumulative=True,):
        
        """ ADMM algorithm for solving general minimization problem
        
                    min_x 1/2*||F(x)||^2,
            
            where F(x) := |A(x)| - B.
            Adjusted from https://gitlab.xlim.fr/shpakovych/phrt-opt
        """
        if isinstance(rho, (float, int)):
            strategy = lambda it: rho
        elif callable(rho):
            strategy = lambda it: rho(it)
        else:
            raise ValueError(f"Parameter 'rho' must be either a number or a callable, but found '{type(rho)}'.")
        
        x = self.X_init
        if x is None:
            x = np.random.randn(self.A.shape[1], self.B.shape[1]) + 1j * np.random.randn(self.A.shape[1], self.B.shape[1])
        x = torch.from_numpy(x.astype(self.dtype)).to(self.device)
        x = nn.Parameter(x, requires_grad=True)

        A = torch.from_numpy(self.A.astype(dtype)).to(self.device)
        B = torch.from_numpy(self.B).to(self.device)
        B = B.to(A.device)
        B_norm = torch.norm(B)

        optimizer = torch.optim.Adam([x], lr=lr)
        lmd = np.zeros(tuple(B.shape), dtype=dtype)
        lmd = torch.from_numpy(lmd).to(self.device)
        prev_dist = None
        best_mse = None
        intens_mse = None

        # ADMM optimization loop to minimize f(x) := ||F(x)||^2
        for it in range(max_iter):
            _rho = strategy(it)

            # x_{k+1} = argmin_x ||F(x) - (1 - rho)lmd_{k}||^2
            for _ in range(max_inner_iter):
                optimizer.zero_grad()
                loss = torch.norm(torch.abs(torch.matmul(A, x)) - B + (1 - _rho) * lmd)
                loss.backward()
                optimizer.step()
            # lmd_{k+1} = (1 / (1 + rho)) * F(x)
            with torch.no_grad():
                lmd = (1. / (1 + _rho)) * (lmd * int(is_lmd_cumulative) + torch.matmul(A, x) - B)

            # Stopping test
            dist = torch.norm(torch.abs(torch.matmul(A, x)) - B) / B_norm
            if prev_dist is None:
                prev_dist = dist
                continue
            stopping_dist = torch.abs(dist - prev_dist)
            if self.verbose > 0:
                print(f"Iteration: {it:3d} - distance = {dist:.3e} - stopping distance = {stopping_dist:.3e}")
            if stopping_dist < tol:
                if self.verbose > 0:
                    print(f"Convergence detected: {stopping_dist:.3e} < {tol:.2e} ")
                break
            prev_dist = dist

            intens_mse = torch.mean(torch.abs(torch.square(torch.abs(torch.matmul(A, x))) - torch.square(B)))
            if best_mse is None:
                best_mse = intens_mse
            if intens_mse < best_mse:
                best_mse = intens_mse
                self.X_best = x.cpu().detach().numpy()

            self.iter.append(it)
            self.metric.append(dist.cpu().detach().numpy())
            self.mse.append(intens_mse.cpu().detach().numpy())
        self.X = x.cpu().detach().numpy()

    @property
    def X_norm(self):
        return TMR.normalize_matrix(self.X)
    
    @property
    def X_best_norm(self):
        return TMR.normalize_matrix(self.X_best)

    @staticmethod
    def normalize_matrix(X):
        return np.abs(X) / np.max(np.abs(X)) * np.exp(1j * (np.angle(X) - np.angle(X[0, :])))

    def show_results(self):
        x = np.array(self.iter)+1
        y1 = np.array(self.metric)
        y2 = np.array(self.mse)

        fig, ax = plt.subplots()
        ln1 = ax.plot(x, y1, color='blue', label='Metric')
        ax.set_xlabel('Iteration #')
        ax.set_ylabel('Distance [a.u.]', color='blue')
        ax.set_title('ADMM convergence')
        
        ax2 = ax.twinx()
        ln2 = ax2.plot(x, y2, color='red', label='Intensity MSE')
        ax2.set_ylabel('MSE',color='red')

        leg = ln1 + ln2
        labs = [l.get_label() for l in leg]
        ax.legend(leg, labs)