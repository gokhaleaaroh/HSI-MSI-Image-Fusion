import numpy as np
import jax.numpy as jnp
from jax import jit, value_and_grad
from scipy.optimize import minimize
import torch

from motion_code.sparse_gp import *
from motion_code.utils import *

# from sparse_gp import *
# from utils import *


def optimize_motion_codes(X_list, Y_list, labels, model_path, m=10, Q=8, latent_dim=3, sigma_y=0.1):
    '''
    Main algorithm to optimize all variables for the Motion Code model.
    '''
    num_motion = np.unique(labels).shape[0]
    dims = (num_motion, m, latent_dim, Q)

    # Initialize parameters
    X_m_start = np.repeat(sigmoid_inv(np.linspace(0.1, 0.9, m)).reshape(1, -1), latent_dim, axis=0).swapaxes(0, 1)
    Z_start = np.ones((num_motion, latent_dim))
    Sigma_start = softplus_inv(np.ones((num_motion, Q)))
    W_start = softplus_inv(np.ones((num_motion, Q)))

    # Optimize X_m, Z, and kernel parameters including Sigma, W
    # res = minimize(fun=elbo_fn_vec(X_list, Y_list, labels, sigma_y, dims),
    #     x0 = pack_params([X_m_start, Z_start, Sigma_start, W_start]),
    #     method='L-BFGS-B', jac=True)


    def callback_function(xk):
        """
        Called after each iteration.
        xk is the current parameter vector.
        """
        print(f"Iteration: {callback_function.iteration}")
        callback_function.iteration += 1
 
    callback_function.iteration = 0

    res = minimize(
        fun=elbo_fn(X_list, Y_list, labels, sigma_y, dims),
        x0=pack_params([X_m_start, Z_start, Sigma_start, W_start]),
        method='L-BFGS-B',
        jac=True, callback=callback_function
    )

    print(res.message)

    # res = minimize(
    #     fun=elbo_fn(X_list, Y_list, labels, sigma_y, dims),
    #     x0=pack_params([X_m_start, Z_start, Sigma_start, W_start]),
    #     method='L-BFGS-B',
    #     jac=True,
    #     options={'maxcor': 100, 'ftol': 1e-5, 'gtol': 1e-5}
    # )


    X_m, Z, Sigma, W = unpack_params(res.x, dims=dims)
    Sigma = softplus(Sigma)
    W = softplus(W)

    # We now optimize distribution params for each motion and store means in mu_ms, covariances in A_ms, and for convenient K_mm_invs
    mu_ms = []; A_ms = []; K_mm_invs = []

    # All timeseries of the same motion is put into a list, an element of X_motion_lists and Y_motion_lists
    X_motion_lists = []; Y_motion_lists = []
    for _ in range(num_motion):
        X_motion_lists.append([]); Y_motion_lists.append([])
    for i in range(len(Y_list)):
        X_motion_lists[labels[i]].append(X_list[i])
        Y_motion_lists[labels[i]].append(Y_list[i])

    # For each motion, using trained kernel parameter in "pair" form to obtain optimal distribution params for each motion.
    for k in range(num_motion):
        kernel_params = (Sigma[k], W[k])
        mu_m, A_m, K_mm_inv = phi_opt(sigmoid(X_m@Z[k]), X_motion_lists[k], Y_motion_lists[k], sigma_y, kernel_params) 
        mu_ms.append(mu_m); A_ms.append(A_m); K_mm_invs.append(K_mm_inv)
    
    # Save model to path.
    model = {'X_m': X_m, 'Z': Z, 'Sigma': Sigma, 'W': W, 
             'mu_ms': mu_ms, 'A_ms': A_ms, 'K_mm_invs': K_mm_invs}
    np.save(model_path, model)
    return

def optimize_motion_codes_gpu(X_list, Y_list, labels, model_path, m=10, Q=8, latent_dim=3, sigma_y=0.1, maxiter=50000, tol=1e-6):
    '''
    GPU-accelerated version of optimize_motion_codes.
    Uses PyTorch's L-BFGS optimizer so the entire optimization loop
    (including L-BFGS state updates and gradient evaluations via autograd)
    stays on the GPU, avoiding per-iteration device round-trips.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    dtype = torch.float64

    num_motion = np.unique(labels).shape[0]
    dims = (num_motion, m, latent_dim, Q)

    # ---- Batched PyTorch re-implementations of sparse_gp helpers ----
    def _spectral_kernel_batched(X1, X2, sigma, alpha):
        X12 = (X1.unsqueeze(2) - X2.unsqueeze(1)).unsqueeze(-1)
        return torch.sum(
            alpha.unsqueeze(1).unsqueeze(2)
            * torch.exp(-0.5 * X12 * sigma.unsqueeze(1).unsqueeze(2) * X12),
            dim=-1,
        )

    jitter_val = torch.tensor(1e-6, device=device, dtype=dtype)

    def _elbo_batched(K_mm, K_mn, y, trace_avg, sigma_y, num_valid, mask):
        K_mn = K_mn * mask
        y = y * mask.transpose(1, 2)

        L = torch.linalg.cholesky(K_mm)
        A = torch.linalg.solve_triangular(L, K_mn, upper=False) / sigma_y
        AAT = torch.matmul(A, A.transpose(-1, -2))

        B = torch.eye(K_mm.shape[1], device=device, dtype=dtype).unsqueeze(0) + AAT
        # Add jitter to B to guarantee it remains strictly positive definite
        # B = B + torch.eye(K_mm.shape[1], device=device, dtype=dtype).unsqueeze(0) * b_jitter_val
        LB = torch.linalg.cholesky(B)

        Ay = torch.matmul(A, y)
        c = torch.linalg.solve_triangular(LB, Ay, upper=False) / sigma_y

        lb = -num_valid / 2 * np.log(2 * np.pi)
        lb = lb - torch.sum(torch.log(torch.diagonal(LB, dim1=-2, dim2=-1)), dim=-1)
        lb = lb - num_valid / 2 * np.log(sigma_y ** 2)

        y_sq = torch.sum(y ** 2, dim=[1, 2])
        c_sq = torch.sum(c ** 2, dim=[1, 2])

        lb = lb - 0.5 / sigma_y ** 2 * y_sq
        lb = lb + 0.5 * c_sq
        lb = lb - 0.5 / sigma_y ** 2 * num_valid * trace_avg

        aat_tr = torch.sum(torch.diagonal(AAT, dim1=-2, dim2=-1), dim=-1)
        lb = lb + 0.5 * aat_tr

        return torch.mean(-lb)

    # ---- Move data to device (Batched and Padded) ----
    n_series = len(X_list)
    max_len = max(len(x) for x in X_list)
    X_pad = np.zeros((n_series, max_len))
    Y_pad = np.zeros((n_series, max_len))
    mask = np.zeros((n_series, max_len))
    num_valid = np.zeros(n_series)

    for i in range(n_series):
        L = len(X_list[i])
        X_pad[i, :L] = np.asarray(X_list[i]).flatten()
        Y_pad[i, :L] = np.asarray(Y_list[i]).flatten()
        mask[i, :L] = 1.0
        num_valid[i] = L

    X_pad_t = torch.tensor(X_pad, device=device, dtype=dtype)
    Y_pad_t = torch.tensor(Y_pad, device=device, dtype=dtype).unsqueeze(-1)
    mask_t = torch.tensor(mask, device=device, dtype=dtype).unsqueeze(1)
    num_valid_t = torch.tensor(num_valid, device=device, dtype=dtype)
    labels_arr = np.asarray(labels)
    labels_t = torch.tensor(labels_arr, device=device, dtype=torch.long)

    # ---- Initialize parameters ----
    X_m_start = np.repeat(
        sigmoid_inv(np.linspace(0.1, 0.9, m)).reshape(1, -1),
        latent_dim, axis=0,
    ).swapaxes(0, 1)
    Z_start = np.ones((num_motion, latent_dim))
    Sigma_start = softplus_inv(np.ones((num_motion, Q)))
    W_start = softplus_inv(np.ones((num_motion, Q)))

    x0 = pack_params([X_m_start, Z_start, Sigma_start, W_start])
    params = torch.tensor(x0, device=device, dtype=dtype, requires_grad=True)

    def _unpack(p):
        idx = 0
        X_m = p[idx:idx + m * latent_dim].reshape(m, latent_dim); idx += m * latent_dim
        Z = p[idx:idx + num_motion * latent_dim].reshape(num_motion, latent_dim); idx += num_motion * latent_dim
        S = p[idx:idx + num_motion * Q].reshape(num_motion, Q); idx += num_motion * Q
        W = p[idx:idx + num_motion * Q].reshape(num_motion, Q)
        return X_m, Z, S, W

    def closure():
        optimizer.zero_grad()
        X_m, Z, S, W = _unpack(params)
        S = torch.nn.functional.softplus(S)
        W = torch.nn.functional.softplus(W)

        # Broadcasted batched parameters
        Z_b = Z[labels_t]
        S_b = S[labels_t]
        W_b = W[labels_t]

        X_m_k_b = torch.sigmoid(torch.matmul(Z_b, X_m.T))
        
        K_mm = _spectral_kernel_batched(X_m_k_b, X_m_k_b, S_b, W_b)
        eye_m = torch.eye(m, device=device, dtype=dtype).unsqueeze(0) * jitter_val
        K_mm = K_mm + eye_m
        
        K_mn = _spectral_kernel_batched(X_m_k_b, X_pad_t, S_b, W_b)
        trace_avg = torch.sum(W_b ** 2, dim=-1)
        
        loss = _elbo_batched(K_mm, K_mn, Y_pad_t, trace_avg, sigma_y, num_valid_t, mask_t)

        loss.backward()
        return loss

    optimizer = torch.optim.LBFGS(
        [params], max_iter=maxiter,
        tolerance_grad=tol, tolerance_change=tol,
        line_search_fn='strong_wolfe',
    )
    optimizer.step(closure)

    # ---- Convert optimized params back to numpy / JAX for phi_opt ----
    params_np = params.detach().cpu().numpy().astype(np.float64)
    X_m, Z, Sigma, W = unpack_params(params_np, dims=dims)
    Sigma = softplus(Sigma)
    W = softplus(W)

    mu_ms = []; A_ms = []; K_mm_invs = []

    X_motion_lists = [[] for _ in range(num_motion)]
    Y_motion_lists = [[] for _ in range(num_motion)]
    for i in range(len(Y_list)):
        X_motion_lists[labels[i]].append(X_list[i])
        Y_motion_lists[labels[i]].append(Y_list[i])

    for k in range(num_motion):
        kernel_params = (Sigma[k], W[k])
        mu_m, A_m, K_mm_inv = phi_opt(sigmoid(X_m @ Z[k]), X_motion_lists[k], Y_motion_lists[k], sigma_y, kernel_params)
        mu_ms.append(mu_m); A_ms.append(A_m); K_mm_invs.append(K_mm_inv)

    model = {'X_m': X_m, 'Z': Z, 'Sigma': Sigma, 'W': W,
             'mu_ms': mu_ms, 'A_ms': A_ms, 'K_mm_invs': K_mm_invs}
    np.save(model_path, model)
    return

def optimize_motion_codes_jax_gpu(X_list, Y_list, labels, model_path, m=10, Q=8, latent_dim=3, sigma_y=0.1, maxiter=500, tol=1e-5):
    '''
    GPU-accelerated version of optimize_motion_codes using JAX.
    Uses jaxopt's L-BFGS optimizer so the entire optimization loop
    (including L-BFGS state updates and gradient evaluations via autodiff)
    stays on the GPU, avoiding per-iteration device round-trips.
    '''
    import jaxopt

    num_motion = np.unique(labels).shape[0]
    dims = (num_motion, m, latent_dim, Q)

    X_list_jax = [jnp.array(x) for x in X_list]
    Y_list_jax = [jnp.array(y) for y in Y_list]
    n_series = len(X_list_jax)
    labels_arr = np.asarray(labels)

    # Initialize parameters
    X_m_start = np.repeat(sigmoid_inv(np.linspace(0.1, 0.9, m)).reshape(1, -1), latent_dim, axis=0).swapaxes(0, 1)
    Z_start = np.ones((num_motion, latent_dim))
    Sigma_start = softplus_inv(np.ones((num_motion, Q)))
    W_start = softplus_inv(np.ones((num_motion, Q)))

    x0 = jnp.array(pack_params([X_m_start, Z_start, Sigma_start, W_start]))

    def elbo(params):
        X_m, Z, Sigma, W = unpack_params(params, dims)
        Sigma = softplus(Sigma)
        W = softplus(W)

        loss = 0.0
        for i in range(n_series):
            k = labels_arr[i]
            X_m_k = sigmoid(X_m @ Z[k])
            K_mm = spectral_kernel(X_m_k, X_m_k, Sigma[k], W[k]) + jitter(X_m_k.shape[0])
            K_mn = spectral_kernel(X_m_k, X_list_jax[i], Sigma[k], W[k])
            trace_avg_all_comps = jnp.sum(W[k] ** 2)
            y_n_k = Y_list_jax[i].reshape(-1, 1)
            loss += elbo_fn_from_kernel(K_mm, K_mn, y_n_k, trace_avg_all_comps, sigma_y)

        return loss / n_series

    solver = jaxopt.LBFGS(fun=elbo, maxiter=maxiter, tol=tol, jit=True)
    result = solver.run(x0)
    params_opt = result.params

    params_np = np.array(params_opt)
    X_m, Z, Sigma, W = unpack_params(params_np, dims=dims)
    Sigma = softplus(Sigma)
    W = softplus(W)

    mu_ms = []; A_ms = []; K_mm_invs = []

    X_motion_lists = [[] for _ in range(num_motion)]
    Y_motion_lists = [[] for _ in range(num_motion)]
    for i in range(len(Y_list)):
        X_motion_lists[labels[i]].append(X_list[i])
        Y_motion_lists[labels[i]].append(Y_list[i])

    for k in range(num_motion):
        kernel_params = (Sigma[k], W[k])
        mu_m, A_m, K_mm_inv = phi_opt(sigmoid(X_m @ Z[k]), X_motion_lists[k], Y_motion_lists[k], sigma_y, kernel_params)
        mu_ms.append(mu_m); A_ms.append(A_m); K_mm_invs.append(K_mm_inv)

    model = {'X_m': X_m, 'Z': Z, 'Sigma': Sigma, 'W': W,
             'mu_ms': mu_ms, 'A_ms': A_ms, 'K_mm_invs': K_mm_invs}
    np.save(model_path, model)
    return

def classify_predict_helper(X_test, Y_test, kernel_params_all_motions, X_m, Z, mu_ms, A_ms, K_mm_invs, mode='dt'):
    """
    Classify by calculate distance between inducing (mean) values and interpolated test values at inducing pts.
    """
    num_motion = len(kernel_params_all_motions)
    ind = -1; min_ll = 1e9
    for k in range(num_motion):
        X_m_k = sigmoid(X_m @ Z[k])
        if mode == 'simple':
            Y = np.interp(X_m_k, X_test, Y_test)
            ll = ((mu_ms[k]-Y).T)@(mu_ms[k]-Y)
        elif mode == 'variational':
            Sigma, W = kernel_params_all_motions[k]
            K_mm = spectral_kernel(X_m_k, X_m_k, Sigma, W) + jitter(X_m_k.shape[0])
            K_mn = spectral_kernel(X_m_k, X_test, Sigma, W)
            trace_avg_all_comps = jnp.sum(W**2)
            y_n_k = Y_test.reshape(-1, 1) # shape (n, 1)
            ll = elbo_fn_from_kernel(K_mm, K_mn, y_n_k, trace_avg_all_comps, sigma_y=0.1)
        elif mode == 'dt':
            mean, _ = q(X_test, X_m_k, kernel_params_all_motions[k], mu_ms[k], A_ms[k], K_mm_invs[k])
            # ll = jnp.log(jnp.linalg.det(covar)) + ((Y_test-mean).T)@jnp.linalg.inv(covar)@(Y_test-mean)
            ll = ((mean-Y_test).T)@(mean-Y_test) 
        if ind == -1:
            ind = k; min_ll = ll
        elif min_ll > ll: 
            ind = k; min_ll = ll
    
    return ind
