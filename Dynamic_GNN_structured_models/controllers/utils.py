import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import trange

def ltv_lqr(x0, F, f, C, c, n, m, T, fwd_pass=False):
    # Backward recursion
    K = np.zeros((T + 1, m, n))
    k = np.zeros((T + 1, m, 1))

    Vtp1 = np.zeros((n, n))
    vtp1 = np.zeros((n, 1))

    Ft = np.zeros((n, n + m))
    ft = np.zeros((n, 1))
    
    for t in range(T, -1, -1):
        Qt = C[t] + Ft.T @ Vtp1 @ Ft
        qt = c[t] + Ft.T @ Vtp1 @ ft + Ft.T @ vtp1

        K[t] = -np.linalg.inv(Qt[n:,n:]) @ Qt[n:,:n]
        k[t] = -np.linalg.inv(Qt[n:,n:]) @ qt[n:,:]
        
        if t > 0:
            Vtp1 = Qt[:n,:n] + Qt[:n, n:] @ K[t] + K[t].T @ Qt[n:, :n] + K[t].T @ Qt[n:,n:] @ K[t]
            vtp1 = qt[:n] + Qt[:n, n:] @ k[t] + K[t].T @ qt[n:, :] + K[t].T @ Qt[n:,n:] @ k[t]
            
            Ft[:] = F[t-1]
            ft[:] = f[t-1]
            
    # Forward pass
    if fwd_pass:
        tau = np.zeros((T+1, n + m, 1))
        xt = np.copy(x0)
        for t in range(T):
            ut = K[t]@xt + k[t]
            tau[t] = np.append(xt, ut, axis = 0) 
            xtp1 = F[t]@tau[t] + f[t]
            xt = np.copy(xtp1)
        tau[T] = np.append(xtp1, np.zeros((m, 1)), axis = 0)

        return K, k, tau
    else:
        return K, k

def ltv_lqr2(x0, xf, Q, R, A, B, n, m, T, fwd_pass=False):
    '''
    Regularization about xf using LQR with discrete-time linear model x(t+1) = A(t)x(t)+B(t)u(t)

    Inputs:
        x0: size = (n, 1)
        xf: size = (n, 1)
        R: size = (T, m, m)
        Q: size = (T, n, n)
        A: size = (T, n, n)
        B: size = (T, n, m)
    Return:
        K: size = (T, m, m)
        k: size = (T, m, 1)
        tau: size = (T, n+m, 1)
    '''

    K = np.zeros((T, m, n))
    k = np.zeros((T, m, 1))

    Stp1 = Q[T-1].copy()
    for t in range(T-2, -1, -1):
        K[t] = -np.linalg.inv(R[t] + B[t].T@Stp1@B[t])@B[t].T@Stp1@A[t]
        k[t] = -K[t]@xf
        Stp1 = Q[t] + A[t].T@np.linalg.inv(np.eye(n) + Stp1@B[t]@np.linalg.inv(R[t])@B[t].T)@Stp1@A[t]
            
    # Forward pass
    if fwd_pass:
        tau = np.zeros((T, n + m, 1))
        xt = np.copy(x0)
        for t in range(T-1):
            ut = K[t]@xt + k[t]
            tau[t] = np.append(xt, ut, axis = 0) 
            xtp1 = A[t]@xt + B[t]@ut
            xt = np.copy(xtp1)
        tau[T-1] = np.append(xtp1, np.zeros((m, 1)), axis = 0)
        return K, k, tau
    else:
        return K, k

def iLQR_torch(z0, zf, T, Q, R, A, B, thresh_limit, n_ilqr_iter, model):
    tol = 100
    thresh = 0.1
    i = 0
    while thresh > thresh_limit and i < n_ilqr_iter:
        if i > 0:
            tol_old = tol
            tau_old = tau.clone()
        
        control = dict()
        control['K'], control['k'] = model.LQR(1, T, A, B, Q, R, z0.pos[None,:,:,None], zf.pos[None,:,:,None])
        tau, A, B, _ = model.forward_propagate_control_lqr(z0, control, return_numpy_dyn=False)
        Q, R = model.quadratize_cost(tau, zf.pos[None,:,:], T)

        if i > 0:
            tol = torch.linalg.norm(tau.pos-tau_old.pos)
            thresh = (tol-tol_old)**2
        i += 1
    # print(f"iLQR iters:{i}")
    return tau, control, i

def iLQR1(x0, xf, n, m, T, F, f, C, c, thresh_limit, n_ilqr_iter, model, env, h0=None, return_mode=False):
    tol = 100
    thresh = 0.1
    i = 0
    while thresh > thresh_limit and i < n_ilqr_iter:
        if i > 0:
            tol_old = tol
            tau_old = tau_batch.clone()
        
        control = dict()
        control['K'], control['k'] = ltv_lqr(x0, F, f, C, c, n, m, T)

        tau_batch, F, f, mode = model.forward_propagate_control_lqr(x0, control,ht=h0,return_mode=return_mode)

        # tau = model._post_process_tau_batch_to_np(tau_batch, model.n)
        # C, c = env.quadratize_cost(tau_batch)

        if i > 0:
            tol = torch.linalg.norm(tau_batch.pos-tau_old.pos)
            # tol = np.linalg.norm(tau_batch-tau_old)
            thresh = (tol-tol_old)**2
        i += 1
    print(f"iLQR iters:{i}")
    return tau_batch, control, i, mode

def iLQR2(x0, n, m, T, F, f, C, c, thresh_limit, n_ilqr_iter, model, env):
    tol = 100
    thresh = 0.1
    i = 0
    while thresh > thresh_limit and i < n_ilqr_iter:
        if i > 0:
            tol_old = tol
            tau_old = tau.copy()
        
        control = dict()
        control['K'], control['k'], tau = ltv_lqr(x0, F, f, C, c, n, m, T, fwd_pass=True)

        F, f, C, c= model.linearize_dynamics(tau, env)

        if i > 0:
            tol = np.linalg.norm(tau-tau_old)
            thresh = (tol-tol_old)**2
        i += 1
    return tau, control, i