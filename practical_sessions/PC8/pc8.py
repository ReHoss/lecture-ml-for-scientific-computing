# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: venv_sci_ml
#     language: python
#     name: python3
# ---

# %% [markdown]
# <div style="text-align: center;">
# <strong>Machine Learning for Scientific Computing and Numerical Analysis - Practical 8</strong>
# </div>
# <div style="text-align: center;">
# <i>Randomisation in Scientific Machine Learning: Sketching, Randomized SVD, Nyström, Random Features</i>
# </div>
# <div style="text-align: center;">
# </br>
# <p>Victorita Dolean, Loïc Gouarin, Rémy Hosseinkhan, Hadrien Montanelli
# </br>
# 2025-2026
# </p>
# </div>
#
# This notebook covers:
# 1. Johnson–Lindenstrauss (JL) random projections
# 2. Sketching for least-squares
# 3. Randomized SVD
# 5. Random Fourier Features (RFF)
# 6. Mini-batching as sketching for PDE solvers

# %% [markdown]
# ---
# # PART 1: Johnson–Lindenstrauss (JL) Random Projections

# %% [markdown]
# # 1 Imports and helper functions

# %%
import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Tuple, Callable

np.random.seed(0)

def rel_err(a, b, eps=1e-12):
    return npla.norm(a-b) / (npla.norm(b) + eps)

def rbf_kernel(X, Y=None, sigma=0.2):
    if Y is None:
        Y = X
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    XX = np.sum(X**2, axis=1)[:,None]
    YY = np.sum(Y**2, axis=1)[None,:]
    D2 = XX + YY - 2*X@Y.T
    return np.exp(-D2/(2*sigma**2))

def stable_svd(A):
    U, s, VT = npla.svd(A, full_matrices=False)
    return U, s, VT

print("numpy", np.__version__)

# %% [markdown]
# # 1 Dimensionality reduction: Johnson–Lindenstrauss (JL)

# %% [markdown]
# We revisit the core message: **random projections preserve geometry**.
#
# Given points $x_1,\dots,x_N\in\mathbb{R}^d$ and a random matrix $S\in\mathbb{R}^{m\times d}$,
# the JL lemma says that for $m=O(\varepsilon^{-2}\log N)$ we have with high probability
#
# $$
# (1-\varepsilon)\|x_i-x_j\|_2^2 \le \|Sx_i-Sx_j\|_2^2 \le (1+\varepsilon)\|x_i-x_j\|_2^2
# \quad \forall i,j.
# $$

# %% [markdown]
# ## 1.1 JL projection and distortion metric

# %% [markdown]
# ### Exercise
#
# Complete the missing parts.

# %%
def jl_project(X: np.ndarray, m: int, kind: str = "gaussian", seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project X (N,d) to (N,m) with a random JL matrix S (m,d).
    """
    rng = np.random.default_rng(seed)
    N, d = X.shape
    S = None
    Z = None
    return Z, S


def max_pairwise_distortion(X: np.ndarray, Z: np.ndarray, n_pairs: int = 2000, seed: int = 0) -> float:
    """
    Estimate max relative distortion over a random subset of pairs.
    Returns $\max_{(i,j)} | \|z_i-z_j\| / \|x_i-x_j\| - 1 |$.
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    distortion = None
    return distortion

# %% [markdown]
# ## 1.2 Empirical JL scaling

# %% [markdown]
# ### Exercise
#
# Complete the missing parts.

# %%
N, d = 600, 200
t = np.linspace(0, 4*np.pi, N)
X = np.zeros((N, d))
X[:,0] = np.cos(t)
X[:,1] = np.sin(t)
X[:,2] = 0.2*np.cos(3*t)
X[:,3] = 0.2*np.sin(3*t)
X += 0.05*np.random.randn(N, d)

ms = [10, 20, 40, 80, 120]
dist_gauss, dist_rade = [], []

for m in ms:
    # TODO: compute max distortion for Gaussian and Rademacher projections
    # dist_gauss.append(...)
    # dist_rade.append(...)
    pass

plt.figure()
plt.semilogy(ms, dist_gauss, marker='o', label="Gaussian")
plt.semilogy(ms, dist_rade, marker='o', label="Rademacher")
plt.xlabel("Projection dimension m")
plt.ylabel("Estimated max distortion")
plt.grid(True)
plt.legend()
plt.show()

# %% [markdown]
# ---
# # PART 2: Sketching Least-Squares

# %% [markdown]
# # 2 Sketching least-squares: subspace embeddings in practice

# %% [markdown]
# The sketch-and-solve theorem says: if $S$ is a $(1\pm\varepsilon)$ subspace embedding for $\mathrm{range}([A\ b])$, then solving the sketched problem
#
# $$
# \min_\theta \|SA\theta - Sb\|_2^2
# $$
#
# yields (almost) the same residual as the full least-squares.

# %% [markdown]
# ## 2.1 Sketch-and-solve implementation

# %% [markdown]
# ### Exercise
#
# Complete the missing parts.

# %%
def sketch_and_solve(A: np.ndarray, b: np.ndarray, m: int, method: str = "gaussian", seed: int = 0) -> np.ndarray:
    """
    Return theta_hat that minimizes $\|SA\theta - Sb\|_2$ with m rows after sketching.
    """
    rng = np.random.default_rng(seed)
    n, p = A.shape
    theta_hat = None
    return theta_hat

# %% [markdown]
# ## 2.2 Compare residuals vs sketch size

# %% [markdown]
# ### Exercise
#
# Complete the missing parts.

# %%
n, p = 4000, 40
U = np.random.randn(n, p)
scales = np.logspace(0, 3, p)
A = U @ np.diag(1/scales)
theta_true = np.random.randn(p)
b = A @ theta_true + 0.01*np.random.randn(n)

theta_full, *_ = npla.lstsq(A, b, rcond=None)
r_full = npla.norm(A@theta_full - b)

ms = [100, 200, 400, 800, 1200]
err_gauss, err_rows = [], []

for m in ms:
    # TODO: compute relative residual increase compared to full solution
    # theta_g = sketch_and_solve(...)
    # err_gauss.append(...)
    # theta_r = sketch_and_solve(...)
    # err_rows.append(...)
    pass

plt.figure()
plt.semilogy(ms, err_gauss, marker='o', label="Gaussian sketch")
plt.semilogy(ms, err_rows, marker='o', label="Row sampling")
plt.xlabel("Sketch size m")
plt.ylabel("Relative residual increase")
plt.grid(True)
plt.legend()
plt.show()

# %% [markdown]
# ---
# # PART 3: Randomized Low-Rank Approximation

# %% [markdown]
# # 3 Randomized SVD

# %% [markdown]
# The randomized SVD (rSVD) builds a basis for the dominant column space from a sketch $Y=A\Omega$.

# %% [markdown]
# ## 3.1 Randomized SVD implementation

# %% [markdown]
# ### Exercise
#
# Complete the missing parts.

# %%
def randomized_svd(A: np.ndarray, k: int, p: int = 10, q: int = 0, seed: int = 0):
    """
    Compute a rank-k randomized SVD approximation.
    """
    rng = np.random.default_rng(seed)
    n, d = A.shape
    ell = k + p
    return None

# %% [markdown]
# ## 3.2 Compare rSVD with truncated SVD

# %% [markdown]
# ### Exercise
#
# Complete the missing parts.

# %%
n, d = 600, 500
k_true = 20
U0, _ = npla.qr(np.random.randn(n, k_true))
V0, _ = npla.qr(np.random.randn(d, k_true))
sing = np.linspace(20, 1, k_true)
A = U0 @ np.diag(sing) @ V0.T + 0.05*np.random.randn(n, d)

U, s, VT = stable_svd(A)

ks = [5, 10, 20, 40, 60]
err_tsvd, err_rsvd0, err_rsvd2 = [], [], []

for k in ks:
    # TODO: compute spectral error ||A - A_k||_2 for:
    # - truncated SVD
    # - rSVD with q=0
    # - rSVD with q=2 (power iterations)
    pass

plt.figure()
plt.semilogy(ks, err_tsvd, marker='o', label="Truncated SVD (optimal)")
plt.semilogy(ks, err_rsvd0, marker='o', label="rSVD q=0")
plt.semilogy(ks, err_rsvd2, marker='o', label="rSVD q=2")
plt.xlabel("Rank k")
plt.ylabel(r"Spectral error $\|A-A_k\|_2$")
plt.grid(True)
plt.legend()
plt.show()

# %% [markdown]
# ---
# # PART 5: Random Fourier Features (RFF)

# %% [markdown]
# # 5 Random Fourier Features for RBF kernels

# %% [markdown]
# For a Gaussian RBF kernel
#
# $$
# k(x,y) = \exp\!\left(-\frac{\|x-y\|^2}{2\sigma^2}\right),
# $$

# %% [markdown]
# ## 5.1 RFF feature map implementation

# %% [markdown]
# ### Exercise
#
# Complete the missing parts.

# %%
@dataclass
class RFF:
    W: np.ndarray
    b: np.ndarray

def rff_fit(m: int, d: int, sigma: float, seed: int = 0) -> RFF:
    """
    Sample frequencies for the RBF kernel with lengthscale sigma.
    """
    rng = np.random.default_rng(seed)
    W = None
    b = None
    return RFF(W=W, b=b)

def rff_transform(rff: RFF, X: np.ndarray) -> np.ndarray:
    """
    Return $\Phi(X)$ in $\mathbb{R}^{n \times m}$.
    """
    Phi = None
    return Phi

# %% [markdown]
# ## 5.2 Kernel approximation error and regression

# %% [markdown]
# ### Exercise
#
# Complete the missing parts.

# %%
n = 800
x = np.linspace(0, 1, n)[:,None]
y = np.sin(2*np.pi*x[:,0]) + 0.3*np.sin(8*np.pi*x[:,0])

idx = np.random.permutation(n)
ntr = int(0.7*n)
tr, te = idx[:ntr], idx[ntr:]
xtr, ytr = x[tr], y[tr]
xte, yte = x[te], y[te]

sigma = 0.12
lam = 1e-6

ms = [50, 100, 200, 400, 800]
kerr, test_mse = [], []

for m in ms:
    pass

plt.figure()
plt.semilogy(ms, kerr, marker='o')
plt.xlabel("Number of random features m")
plt.ylabel("Kernel approx error (subset, rel. Frobenius)")
plt.grid(True)
plt.show()

plt.figure()
plt.semilogy(ms, test_mse, marker='o')
plt.xlabel("Number of random features m")
plt.ylabel("Test MSE")
plt.grid(True)
plt.show()

# %% [markdown]
# ---
# # PART 6: Mini-Batching as Sketching for PDE Solvers

# %% [markdown]
# # 6 Toy PDE solver: random features + residual sketching

# %% [markdown]
# We illustrate the "PINN as least-squares" viewpoint with a **linear** random-feature ansatz.

# %% [markdown]
# ## 6.1 Build residual matrix for the Poisson RF model

# %% [markdown]
# ### Exercise
#
# Complete the missing parts.

# %%
def rf_features_and_second_derivative(x: np.ndarray, rff: RFF, activation: str = "tanh"):
    """
    Return $\Phi(x)$ and $\Phi''(x)$ so that $u(x)=\Phi w$ and $u''(x)=\Phi'' w$.
    """
    x = x.reshape(-1, 1)
    if activation == "tanh":
        Phi = None
        Phi_dd = None
        return Phi, Phi_dd
    else:
        raise ValueError("Only 'tanh' implemented.")


def build_poisson_ls_system(N: int, rff: RFF, f: Callable[[np.ndarray], np.ndarray]):
    """
    Return A,b for least-squares enforcing PDE at N interior points and BCs at 0,1.
    """
    x_int = np.linspace(0, 1, N+2)[1:-1]
    A = None
    b = None
    return x_int, A, b

# %% [markdown]
# ## 6.3 Mini-batch / sketching SGD for the same LS problem

# %% [markdown]
# ### Exercise
#
# Complete the missing parts.

# %%
def sgd_least_squares(A: np.ndarray, b: np.ndarray, batch_size: int, n_epochs: int, lr: float, seed: int = 0):
    """
    SGD on $0.5\|Aw-b\|^2$ using batches of rows (row-sketching).
    """
    rng = np.random.default_rng(seed)
    n, m = A.shape
    w = np.zeros(m)
    hist = []
    for ep in range(n_epochs):
        idx = rng.integers(0, n, size=batch_size)
        Ab = A[idx]
        bb = b[idx]
        pass
        r = A @ w - b
        hist.append(0.5*np.dot(r, r))
    return w, np.array(hist)
