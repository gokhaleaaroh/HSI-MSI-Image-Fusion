# Comparison: `optimize_motion_codes` vs `optimize_motion_codes_gpu`

Both functions live in `motion_code/motion_code_utils.py` and solve the same
problem — optimising the parameters of a Motion Code model (inducing points
\(X_m\), motion codes \(Z\), and spectral-kernel hyperparameters \(\Sigma, W\))
by maximising a variational ELBO, then computing the optimal variational
distribution parameters (\(\mu_m, A_m, K_{mm}^{-1}\)) per motion class.
The resulting model dict is saved to disk with `np.save`.

---

## 1. Function Signatures

```python
def optimize_motion_codes(
    X_list, Y_list, labels, model_path,
    m=10, Q=8, latent_dim=3, sigma_y=0.1
)

def optimize_motion_codes_gpu(
    X_list, Y_list, labels, model_path,
    m=10, Q=8, latent_dim=3, sigma_y=0.1,
    maxiter=500, tol=1e-5                     # <-- extra kwargs
)
```

The GPU variant exposes two additional parameters:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `maxiter` | 500 | Maximum number of L-BFGS iterations |
| `tol` | 1e-5 | Convergence tolerance for gradient and parameter change |

In the CPU version, these settings are baked into SciPy's `minimize` defaults
(the L-BFGS-B defaults: `maxiter=15000`, `ftol/gtol=1e-5` for 64-bit).

---

## 2. Frameworks and Device Placement

| Aspect | `optimize_motion_codes` | `optimize_motion_codes_gpu` |
|--------|--------------------------|------------------------------|
| Core framework | **JAX** (via `jax.numpy`, `jax.jit`, `jax.value_and_grad`) | **PyTorch** (`torch`, `torch.nn.functional`, `torch.optim`) |
| Kernel / ELBO code | Re-uses the shared JAX functions in `sparse_gp.py` (`spectral_kernel`, `elbo_fn_from_kernel`, `elbo_fn`) | Provides **self-contained PyTorch re-implementations** (`_spectral_kernel_batched`, `_elbo_batched`) defined as closures inside the function |
| Device | CPU only (SciPy's `minimize` operates on NumPy arrays on CPU) | GPU if available, CPU fallback (`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`) |
| Precision | 64-bit (JAX `x64` enabled globally in `sparse_gp.py`) | 64-bit (`dtype = torch.float64`) |

---

## 3. Optimizer

| Aspect | CPU | GPU |
|--------|-----|-----|
| Library | `scipy.optimize.minimize` | `torch.optim.LBFGS` |
| Method | `L-BFGS-B` (bounded L-BFGS) | L-BFGS with **strong Wolfe** line search |
| Gradient source | JAX `jit(value_and_grad(elbo))` wrapped to return NumPy arrays; passed to SciPy as `jac=True` | PyTorch autograd via `loss.backward()` inside a `closure()` |
| Call pattern | `minimize(fun=..., x0=..., method='L-BFGS-B', jac=True)` — SciPy drives the loop | `optimizer.step(closure)` — PyTorch drives the loop internally |
| Iteration control | SciPy defaults (`maxiter`, `ftol`, `gtol`) | Explicit `max_iter=maxiter`, `tolerance_grad=tol`, `tolerance_change=tol` |

A key performance distinction: in the CPU path every iteration involves a
round-trip between JAX's XLA-compiled ELBO/gradient and SciPy's Python-level
L-BFGS state update. In the GPU path the entire L-BFGS loop (including line
search evaluations and gradient steps) stays on the GPU inside
`torch.optim.LBFGS.step(closure)`.

---

## 4. ELBO Computation — Sequential vs Batched

This is the most significant algorithmic difference between the two
implementations.

### CPU — Sequential per-series loop

The ELBO is computed inside `elbo_fn` (in `sparse_gp.py`):

```python
for i in range(len(X_list)):
    k = labels[i]
    X_m_k = sigmoid(X_m @ Z[k])
    K_mm = spectral_kernel(X_m_k, X_m_k, Sigma[k], W[k]) + jitter(...)
    K_mn = spectral_kernel(X_m_k, X_list[i], Sigma[k], W[k])
    ...
    loss += elbo_fn_from_kernel(K_mm, K_mn, ...)
return loss / len(X_list)
```

Each time series is processed **one at a time** through a Python `for` loop.
JAX's JIT compiles the full loop into a single XLA program, but the loop body
still iterates sequentially over series.

### GPU — Vectorised batched computation

The GPU version:

1. **Pads** all time series to the same length and stacks them into a single
   `(n_series, max_len)` tensor.
2. Creates a binary **mask** `(n_series, max_len)` to ignore padding.
3. Indexes per-motion kernel parameters via label indices
   (`Z_b = Z[labels_t]`, `S_b = S[labels_t]`, etc.), broadcasting them across
   the batch dimension.
4. Computes `K_mm` and `K_mn` for **all series simultaneously** using
   `_spectral_kernel_batched`, which operates on 3D tensors
   `(n_series, m, m)` / `(n_series, m, max_len)`.
5. The batched ELBO (`_elbo_batched`) performs Cholesky, triangular solves,
   and trace/dot operations with batch dimensions, returning
   `torch.mean(-lb)`.

This replaces the per-series Python loop with a single set of batched linear
algebra calls, which is far more efficient on GPUs (and modern CPUs).

---

## 5. Kernel Implementation

### CPU (`spectral_kernel` in `sparse_gp.py`)

```python
X12 = (X1.reshape(num_x1, 1) - X2.reshape(1, num_x2)).reshape(num_x1, num_x2, 1)
return jnp.sum(alpha * jnp.exp(-0.5 * X12 * sigma * X12), axis=-1)
```

Operates on 1D inputs (single series at a time), producing a 2D kernel matrix.

### GPU (`_spectral_kernel_batched`)

```python
X12 = (X1.unsqueeze(2) - X2.unsqueeze(1)).unsqueeze(-1)
return torch.sum(
    alpha.unsqueeze(1).unsqueeze(2)
    * torch.exp(-0.5 * X12 * sigma.unsqueeze(1).unsqueeze(2) * X12),
    dim=-1,
)
```

Operates on a **batch of inputs** — `X1` has shape `(batch, n1)` and the output
is `(batch, n1, n2)`. The kernel parameters `alpha` and `sigma` are also
batched along the first axis.

Both kernels compute the same spectral mixture:
\(\displaystyle K(x_1, x_2) = \sum_{q=1}^{Q} \alpha_q \exp\!\bigl(-\tfrac{1}{2}\sigma_q (x_1 - x_2)^2\bigr)\)

---

## 6. ELBO Formulation Differences

Both compute the same sparse-GP ELBO, but the batched GPU version needs extra
care:

| Detail | CPU (`elbo_fn_from_kernel`) | GPU (`_elbo_batched`) |
|--------|----------------------------|------------------------|
| Masking | Not needed (each series has its true length) | Multiplies `K_mn` and `y` by a binary mask to zero out padding contributions |
| Normalisation constant | `-n/2 * log(2π)` where `n` is the actual length of the series | `-num_valid / 2 * log(2π)` where `num_valid` is a per-series tensor of true lengths |
| Aggregation | Sum over series, divide by `len(X_list)` | `torch.mean(-lb)` over the batch |
| Trace term | `jnp.trace(AAT)` (full trace of the m×m matrix) | `torch.sum(torch.diagonal(AAT, ...))` (batched diagonal-then-sum) |

---

## 7. Data Preprocessing

### CPU

No preprocessing. `X_list` and `Y_list` are passed directly as Python lists of
JAX/NumPy arrays with potentially different lengths.

### GPU

Before optimisation, the function:

1. Finds `max_len = max(len(x) for x in X_list)`.
2. Creates zero-padded NumPy arrays `X_pad`, `Y_pad` of shape
   `(n_series, max_len)`.
3. Builds a mask array of the same shape (1 for valid, 0 for padding).
4. Records `num_valid[i] = len(X_list[i])` for each series.
5. Converts everything to PyTorch tensors on the target device.

This upfront cost enables fully batched computation later.

---

## 8. Parameter Unpacking

### CPU

Uses the shared `unpack_params` from `sparse_gp.py`, which returns JAX arrays.

### GPU

Defines a local `_unpack(p)` closure that slices a flat PyTorch tensor into
`X_m`, `Z`, `S`, `W` with `torch.Tensor.reshape`. This is necessary because the
parameters live as a single `torch.Tensor` with `requires_grad=True` for
autograd.

After optimisation, the GPU version converts back to NumPy
(`params.detach().cpu().numpy().astype(np.float64)`) and calls the shared
`unpack_params` to recover the same format.

---

## 9. Post-Optimisation (Shared Logic)

After the main optimisation loop, **both functions perform identical steps**:

1. Apply `softplus` to `Sigma` and `W` to get the final kernel parameters.
2. Group time series by motion label into `X_motion_lists` /
   `Y_motion_lists`.
3. For each motion `k`, call `phi_opt(sigmoid(X_m @ Z[k]), ...)` to compute
   `mu_m`, `A_m`, `K_mm_inv`.
4. Package everything into a model dict and save via `np.save`.

The `phi_opt` call and all downstream prediction code use JAX in both cases.

---

## 10. Numerical Stability

| Technique | CPU | GPU |
|-----------|-----|-----|
| Jitter on \(K_{mm}\) | `jitter(d, value=1e-6)` — adds \(10^{-6} I\) | `eye_m * jitter_val` where `jitter_val = 1e-6` — same value |
| Jitter on \(B = I + AA^T\) | Not applied (relies on JAX Cholesky) | Commented-out line exists (`b_jitter_val`), currently disabled |
| Softplus for positivity | `jnp.log(1 + exp(x))` (JAX) | `torch.nn.functional.softplus` (PyTorch) |

---

## 11. Summary Table

| Feature | `optimize_motion_codes` | `optimize_motion_codes_gpu` |
|---------|--------------------------|------------------------------|
| Framework | JAX + SciPy | PyTorch |
| Device | CPU only | GPU (with CPU fallback) |
| ELBO loop | Sequential Python for-loop (JIT-compiled) | Fully batched tensor ops |
| Data layout | Variable-length lists | Padded + masked tensors |
| Optimizer | SciPy L-BFGS-B | PyTorch L-BFGS (strong Wolfe) |
| Gradient computation | `jax.value_and_grad` | `torch.autograd` (`loss.backward()`) |
| Iteration control | SciPy defaults | Explicit `maxiter` / `tol` args |
| Kernel code | Shared `spectral_kernel` | Local `_spectral_kernel_batched` |
| ELBO code | Shared `elbo_fn` / `elbo_fn_from_kernel` | Local `_elbo_batched` |
| Post-optimisation | Shared `phi_opt` (JAX) | Same — converts back to NumPy/JAX |
| Output format | Identical model dict saved via `np.save` | Identical model dict saved via `np.save` |
