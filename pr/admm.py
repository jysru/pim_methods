import numpy as np
import torch
import torch.nn as nn

# Adjusted from https://gitlab.xlim.fr/shpakovych/phrt-opt


def admm_pr(
    A,
    b,
    x0=None,
    rho=0.25,
    max_iter=100,
    max_inner_iter=100,
    tol=1e-11,
    lr=1e-2,
    dtype='complex64',
    is_lmd_cumulative=True,
    verbose=0,
    device='cpu',
):  
    """ ADMM algorithm for solving general minimization problem
     
                min_x 1/2*||F(x)||^2,
        
        where F(x) := |A(x)| - b.
    """
    if isinstance(rho, (float, int)):
        strategy = lambda it: rho
    elif callable(rho):
        strategy = lambda it: rho(it)
    else:
        raise ValueError(f"Parameter 'rho' must be either a number or a callable, but found '{type(rho)}'.")
    

    x = x0
    if x is None:
        x = np.random.randn(A.shape[1],1) + 1j * np.random.randn(A.shape[1],1)
    x = torch.from_numpy(x.astype(dtype)).to(device)
    x = nn.Parameter(x, requires_grad=True)

    A = torch.from_numpy(A.astype('complex64')).to(device)
    b = torch.from_numpy(b).to(device)
    b = b.to(device)
    b_norm = torch.norm(b)

    optimizer = torch.optim.Adam([x], lr=lr)

    lmd = np.zeros(tuple(b.shape), dtype=dtype)
    lmd = torch.from_numpy(lmd).to(A.device)
    
    prev_dist = None

    # ADMM optimization loop to minimize f(x) := ||F(x)||^2
    for it in range(max_iter):
        _rho = strategy(it)

        # x_{k+1} = argmin_x ||F(x) - (1 - rho)lmd_{k}||^2
        for _ in range(max_inner_iter):
            optimizer.zero_grad()
            loss = torch.norm(torch.abs(torch.matmul(A, x)) - b + (1 - _rho) * lmd)
            loss.backward()
            optimizer.step()
        # lmd_{k+1} = (1 / (1 + rho)) * F(x)
        with torch.no_grad():
            lmd = (1. / (1 + _rho)) * (lmd * int(is_lmd_cumulative) + torch.matmul(A, x) - b)

        # Stopping test
        dist = torch.norm(torch.abs(torch.matmul(A, x)) - b) / b_norm
        if prev_dist is None:
            prev_dist = dist
            continue
        stopping_dist = torch.abs(dist - prev_dist)
        if verbose > 0:
            print(f"Iteration: {it:3d} - distance = {dist:.3e} - stopping distance = {stopping_dist:.3e}")
        if stopping_dist < tol:
            if verbose > 0:
                print(f"Convergence detected: {stopping_dist:.3e} < {tol:.2e} ")
            break
        prev_dist = dist     
    return x