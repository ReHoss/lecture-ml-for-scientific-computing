# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: venv_sci_ml
#     language: python
#     name: python3
# ---

# %% [markdown] editable=true slideshow={"slide_type": ""}
# <div style="text-align: center;">
# <strong>Machine Learning for Scientific Computing and Numerical Analysis - Practical 2</strong>
# </div>
# <div style="text-align: center;">
# <i>Neural network approximation theory</i>
# </div>
# <div style="text-align: center;">
# </br>
# <p>Victorita Dolean, Loïc Gouarin, Rémy Hosseinkhan, Hadrien Montanelli
# </br>
# 2025-2026
# </p>

# %% [markdown]
# # Introduction
#
# In this session, we explore the expressive power of neural networks. We will proceed in two parts:
# 1. **Implementation**: Implementing the forward pass of deep networks.
# 2. **Approximation Theory**: Constructing a neural network that approximates functions based on the constructive proof of **Yarotsky's Theorem** (Chapter 2).
#
# We will refer to the exercises from the lecture notes (Chapter 2) throughout the session.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from math import factorial, ceil, log
import time

# Set random seed for reproducibility
np.random.seed(42)


# %% [markdown]
# # 1. Basic Building Blocks
#
# We define the standard activation functions used in the lectures.

# %% [markdown]
# ## Exercise
#
# Implement the `__call__` and `diff` methods for the `sigmoid`, `tanh`, and `ReLU` activation functions. The `__call__` method should compute the activation function, while the `diff` method should compute its derivative.


# %%
class sigmoid:
    def __call__(self, x):
        return None  # To implement

    def diff(self, x):
        return None  # To implement


# %%
class tanh:
    def __call__(self, x):
        return None  # To implement

    def diff(self, x):
        return None  # To implement


# %%
class ReLU:
    def __call__(self, x):
        return None  # To implement

    def diff(self, x):
        return None  # To implement


# %% [markdown]
# ### Validation


# %%
activation_functions = [sigmoid, tanh, ReLU]

# %%
x = np.linspace(-5, 5, 200)
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
for ia, a in enumerate(activation_functions):
    ax[ia].plot(x, a()(x), label="activation function")
    ax[ia].plot(x, a().diff(x), label="derivative")
    ax[ia].set_title(a.__name__)
    ax[ia].legend()

fig.tight_layout()


# %% [markdown]
# We will utilize the dataset of the previous PC and give a function to initialize the weights and the biases of the neural network.


# %%
def get_dataset(datatype, n, d, eta, *, random_gen=np.random.rand, seed=42):
    x = -1 + 2 * random_gen(n, d)
    if datatype == "linear":
        w_linear = np.ones(d)
        y = x @ w_linear + eta * random_gen(n)
    elif datatype == "logist":
        w_logistic = np.ones(d)
        y = 1 / (1 + np.exp(-x @ w_logistic)) + eta * random_gen(n)
    elif datatype == "normal":
        mean = np.zeros(d)
        covariance = np.eye(d) / d
        mvn = multivariate_normal(mean=mean, cov=covariance)
        y = mvn.logpdf(x) + eta * random_gen(n)
    elif datatype == "sphere":
        y = np.linalg.norm(x, axis=1) + eta * random_gen(n)
    y /= np.max(np.abs(y))
    return x, y


# %%
def init(d, d1, L, *, eps=1e-2):
    """
    Initializes the weights and biases of a neural network randomly.

    Parameters
    ----------
    d: int
        The size of one entry of the data set.
    d1: int
        The number of neurons in one layer.
    L: int
        The number of hidden layers.
    eps: double
        The perturbation size.

    Returns
    -------
    weights: list
        A list of weights matrices of sizes [(d, d1), (d1, d1), ..., (d1, 1)].
    biases: list
        A list of biases vectors of sizes [(d1), ..., (1)].
    """
    np.random.seed(42)

    weights = []
    biases = []

    previous_size = d

    for l in range(L):
        biases.append(np.zeros([d1, 1]))
        stddev = np.sqrt(2.0 / (previous_size + d1))
        weights.append(
            np.random.normal(loc=0.0, scale=stddev, size=(d1, previous_size))
        )
        previous_size = d1

    biases.append(np.zeros([1, 1]))
    stddev = np.sqrt(2.0 / (previous_size + 1))
    weights.append(np.random.normal(loc=0.0, scale=stddev, size=(1, previous_size)))

    return weights, biases


# %% [markdown]
# # 2. The Forward Pass
#
# Before studying approximation properties, we implement the mechanism of the neural network itself.
#
# The forward pass is the fundamental operation of a neural network during both training and inference (prediction). It is the process by which input data is transformed into an output.
#
# Modern deep learning frameworks (like PyTorch or TensorFlow) use the forward pass to build a dynamic computational graph.
#
# As data moves forward, the framework records every operation (addition, multiplication, activation).
# This record is essential for the backward pass (backpropagation). To calculate gradients (how much to adjust weights), the chain rule requires the intermediate values calculated during the forward pass.
#
# An artificial neural network $\hat{f}(\cdot;\bm{w}):\R^d\to\R$ is a composition of layers:
#
# $$
# \begin{align}
# & \bm{a}^{(0)}(\bm{x}) = \bm{x}, \\
# & \bm{a}^{(\ell)}(\bm{x}) = \sigma\left(W^{(\ell-1)}\bm{a}^{(\ell-1)}(\bm{x}) + \bm{b}^{(\ell-1)}\right), \quad 1\leq\ell\leq L, \\
# & a^{(L+1)}(\bm{x}) = W^{(L)}\bm{a}^{(L)}(\bm{x}) + b^{(L)}.
# \end{align}
# $$

# %% [markdown]
# ## 2.1 Shallow Forward Pass
#
# We first consider a shallow neural network with one hidden layer ($L=1$) which allows us to understand the basic mechanics of the forward pass.
#
# $$
# \begin{align}
# & \bm{a}^{(1)}(\bm{x}) = \sigma\left(W^{(0)}\bm{x} + \bm{b}^{(0)}\right), \\
# & a^{(2)}(\bm{x}) = W^{(1)}\bm{a}^{(1)}(\bm{x}) + b^{(1)}.
# \end{align}
# $$

# %% [markdown]
# ## Exercise
#
# Implement the forward pass for a shallow neural network.


# %%
def forward_pass(X, weights, biases, activation=sigmoid()):
    """
    Performs a forward pass through a shallow neural network.

    Parameters
    ----------
    X: numpy.ndarray
        Input data of shape (n, d).
    weights: list of numpy.ndarray
        List of weight matrices for each layer.
    biases: list of numpy.ndarray
        List of bias vectors for each layer.
    activation: callable, optional
        Activation function to use. Default is sigmoid().

    Returns
    -------
    Z: list of numpy.ndarray
        List of linear transformation results for each layer.
    A: list of numpy.ndarray
        List of activations for each layer, including the input data as the first element.
    """
    # Hidden layer
    z1 = None  # To implement
    a1 = None  # To implement

    # Output layer
    z2 = None  # To implement
    a2 = None  # To implement

    return [z1, z2], [X.T, a1, a2]


# %% [markdown]
# ### Validation

# %%
# Setup (data):
n = 5
d = 2
eta = 1e-2
X, y = get_dataset("linear", n, d, eta)

# Setup (NN):
d1 = 3
L = 1
weights, biases = init(d, d1, L)

# Forward pass:
Z, A = forward_pass(X, weights, biases)
print(f"Layers: {len(Z)}, Activations: {len(A)}")
print("Output shape:", A[-1].shape)

# %% [markdown]
# ## 2.2 Deep Forward Pass
#
# We now generalize to an arbitrary number of layers. You will see that the implementation is quite similar to the shallow case, with the addition of a loop over the layers.

# %% [markdown]
# ## Exercise
#
# Given the following implementation of the forward pass, fill in the blanks to handle an arbitrary number of layers.


# %%
def forward_pass_deep(X, weights, biases, activation=sigmoid()):
    """
    Performs a forward pass through a deep neural network.

    Parameters
    ----------
    X: numpy.ndarray
        Input data of shape (n, d).
    weights: list of numpy.ndarray
        List of weight matrices for each layer.
    biases: list of numpy.ndarray
        List of bias vectors for each layer.
    activation: callable, optional
        Activation function to use. Default is sigmoid().

    Returns
    -------
    Z: list of numpy.ndarray
        List of linear transformation results for each layer.
    A: list of numpy.ndarray
        List of activations for each layer, including the input data as the first element.
    """
    L = len(weights) - 1

    Z, A = [], [X.T]

    for l in range(1, L + 1):
        z = None  # To implement
        a = None  # To implement
        Z.append(z)
        A.append(a)

    A.append(None)  # To implement

    return Z, A


# %% [markdown]
# ### Validation

# %%
# Setup (data):
n = 5
d = 2
eta = 1e-2
X, y = get_dataset("linear", n, d, eta)

# Setup (NN):
d1 = 3
L = 2
weights, biases = init(d, d1, L)

# Forward pass:
Z, A = forward_pass_deep(X, weights, biases)
print(f"Layers: {len(Z)}, Activations: {len(A)}")
print("Output shape:", A[-1].shape)


# %% [markdown]
# # 3. Approximation Theory
#
# We now focus on a constructive proof of the Universal Approximation Theorem, specifically **Yarotsky's Theorem** (2017).
#
# Unlike classical existential proofs (e.g., Cybenko, Hornik) which rely on functional analysis to show that an approximation *exists*, Yarotsky's approach explicitly constructs the network and provides error bounds based on depth and width. We will implement the components discussed in **Chapter 2** of the lecture notes.

# %% [markdown]
# ## 3.1 Approximating the Square Function
#
# In **Exercise 2 of Chapter 2**, we discussed how to approximate $x^2$.
#
# We define the "tooth" function $g:[0,1]\to[0,1]$:
#
# $$
# g(x) = 2\sigma(x) - 4\sigma(x-1/2), \qquad \sigma(x)=\text{ReLU}(x).
# $$
#
# And its compositions $g_n = g \circ g_{n-1}$.

# %%
sigma = lambda x: np.maximum(0, x)

def g(x):
    return 2 * sigma(x) - 4 * sigma(x - 1 / 2)

N = 1000
x = np.linspace(0, 1, N)
plt.figure()
plt.plot(x, g(x))
plt.title("The tooth function g(x)")


# %% [markdown]
# ## Exercise
#
# - Construct a function that implements $g_n$ for $n\geq1$, with $g_1=g$ and $g_n = g \circ g_{n-1}$ for $n\geq2$.
# - Run the validation test below &mdash; you should obtain the same functions as on page 2 of the lecture notes.


# %%
def g_n(x, n):
    result = None  # To implemenent
    return result


# %% [markdown]
# ### Validation

# %%
x = np.linspace(0, 1, 1000)
nmax = 3
plt.figure()
for n in range(1, nmax + 1):
    plt.plot(x, g_n(x, n), label=f"$g_{n}$")
plt.legend()
plt.title("Sawtooth functions")


# %% [markdown]
# We approximate $f(x)=x^2$ using the series:
#
# $$
# \hat{f}_n(x) = x - \sum_{j=1}^n\frac{g_j(x)}{4^j}.
# $$
#
# This corresponds to the construction in **Lemma 3.2 (Chapter 2)**.
#
# We showed in the lecture (see proof of **Lemma 3.3**) that
#
# $$
# \Vert f - \hat{f}_n\Vert_\infty \leq \frac{1}{3}\frac{1}{4^n}.
# $$
#
# It is actually possible to prove convergence at a faster rate $1/4^{n+1}$.

# %% [markdown]
# ## Exercise
#
# - Construct a function that implements $\hat{f}_n$ for $n\geq1$.
# - Run the validation test below &mdash; you should obtain convergence at the theoretical sharp rate $1/4^{n+1}$.


# %%
def f_n(x, n):
    result = None  # To implement
    return x - result


# %% [markdown]
# ### Validation

# %%
# Exact square:
f = lambda x: x**2

# Grid for computing errors and plotting:
x = np.linspace(0, 1, 1000)

# Plot f_n and error for increasing n:
nmax = 4
nn = np.arange(1, nmax + 1)
e = np.zeros(nmax)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for n in range(1, nmax + 1):
    axs[0].plot(x, f_n(x, n), label=f"$f_{n}$")
    e[n - 1] = np.max(np.abs(f(x) - f_n(x, n)))

axs[0].plot(x, f(x), "--k", label="$f$")
axs[0].set_xlabel("$x$")
axs[0].legend()
axs[1].semilogy(nn, e, ".-r", label="numerical error between $f$ and $f_n$")
axs[1].semilogy(nn, 1 / 4 ** (nn + 1), "--k", label="theoretical error $1/4^{n+1}$")
axs[1].set_xlabel("$n$")
plt.legend()


# %% [markdown]
# ## 3.2 Multiplication via Polarization
#
# Using the identity $xy = \frac{1}{2}((x+y)^2 - x^2 - y^2)$ (or similar variations), we can approximate multiplication using our square approximation.

# %% [markdown]
# ## Exercise
#
# - Construct a function that implements the approximate product $xy$ using $\hat{f}_n$ via
#
# $$
# \widehat{\Pi}(x,y) = \hat{f}_n\left(\frac{\sigma(x+y)}{2} + \frac{\sigma(-x-y)}{2}\right) - \hat{f}_n\left(\frac{\sigma(x-y)}{2} + \frac{\sigma(y-x)}{2}\right).
# $$
# - Run the validation test bellow &mdash; it should display True.


# %%
def product(x, y, n):
    return None  # To implement


# %% [markdown]
# ### Validation

# %%
x = np.linspace(-1, 1, 1000)
X, Y = np.meshgrid(x, x)
Z = X * Y

epsilon = 1e-5
n = int(np.ceil(np.log(2 / epsilon) / np.log(4) - 1))
np.max(np.abs(Z - product(X, Y, n))) < epsilon

# %% [markdown]
# ## 3.3 Partition of Unity
#
# As detailed in **Exercise 3 of Chapter 2**, we construct a partition of unity using "hat" functions $\phi_\ell$.
#
# For $L\geq1$, define
#
# $$
# \phi_\ell(x) = \psi\Big(3L\Big(x-\frac{\ell}{L}\Big)\Big), \qquad \psi(x) = \sigma(x+2) + \sigma(x-2) - \sigma(x+1) - \sigma(x-1), \qquad 1\leq\ell\leq L.
# $$

# %% [markdown]
# ## Exercise
#
# - Construct a function that implements $\phi_\ell$.
# - Run the validation test below &mdash; you should obtain the same functions as on page 5 of the lecture notes.

# %%
psi = lambda x: sigma(x + 2) + sigma(x - 2) - sigma(x + 1) - sigma(x - 1)


def phi_l(x, l, L):
    return None  # To implement


# %% [markdown]
# ### Validation

# %%
x = np.linspace(0, 1, 1000)
L = 3
plt.figure()
for l in range(L + 1):
    plt.plot(x, phi_l(x, l, L))
plt.title(f"Partition of Unity (L={L})")


# %% [markdown]
# ## 3.4 Local Taylor Polynomials (Theoretical)
#
# We first combine the partition of unity and exact Taylor polynomials.
#
# For $s\geq1$ and $L\geq 1$, define
#
# $$
# f_L(x) = \sum_{\ell=0}^L \phi_\ell(x)P_\ell(x) \qquad \text{with} \qquad P_\ell(x) = \sum_{j=0}^{s-1} \frac{f^{(j)}(\frac{\ell}{L})}{j!}\Big(x-\frac{\ell}{L}\Big)^j.
# $$
#
# We showed in the lecture that
#
# $$
# \Vert f - f_L\Vert \leq \frac{2}{s!}\left(\frac{1}{L}\right)^s,
# $$
#
# for functions in $W^{s,\infty}([0,1])$.

# %% [markdown]
# ## Exercise
#
# - Construct a function that implements $f_L$ for functions $f$ of the form
#
# $$
# f(x) = \vert x - 0.5\vert^s,
# $$
#
# for some $s\geq 1$. We note that, for odd $s$, these functions $f$ are in $W^{s,\infty}([0,1])$.
# - Run the validation code &mdash; you should obtain convergence at the theoretical rate $2/s!(1/L)^s$.


# %%
def f_L(x, L, s):
    result = None  # To implement
    return result


# %% [markdown]
# ### Validation

# %%
# Grid for computing errors and plotting:
x = np.linspace(0, 1, 1000)
s = 5
f = lambda x: np.abs(x - 0.5) ** s

Lmax = 12
LL = np.arange(1, Lmax + 1)
e = np.zeros(Lmax)

for L in LL:
    e[L - 1] = np.max(np.abs(f(x) - f_L(x, L, s)))

plt.figure()
plt.loglog(LL, e, ".-r", label="Numerical error")
plt.loglog(LL, 2 / factorial(s) * 1 / LL**s, "--k", label="Theoretical error")
plt.xlabel("L")
plt.ylabel("Error")
plt.legend()
plt.show()


# %% [markdown]
# ## 3.5 Yarotsky's Network (Full Approximation)
#
# We now replace the exact multiplications in $f_L$ with our approximate product network $\widehat{\Pi}$.
#
# We define our neural network approximation via
#
# $$
# \hat{f}_L(x) = \sum_{\ell=0}^L \widehat{\Pi}(\phi_l(x),\widehat{P}_l(x)) \qquad \text{with} \qquad \widehat{P}_\ell(x) = \sum_{j=0}^{s-1} \frac{f^{(j)}(\frac{\ell}{L})}{j!}\widehat{\Pi}\Big(\underbrace{x-\frac{\ell}{L},\ldots,x-\frac{\ell}{L}}_{\text{$j$ times}}\Big).
# $$

# %% [markdown]
# ## Exercise
#
# - Construct a function that implements $\hat{f}_L$ for functions $f$ of the form
#
# $$
# f(x) = \vert x - 0.5\vert^s, \qquad s\geq 1.
# $$
#
# - Run the validation code &mdash; you should obtain convergence at the same theoretical rate $2/s!(1/L)^s$


# %%
def fhat_L(x, L, s, n):
    result = None  # To implement
    return result


# %% [markdown]
# ### Validation

# %%
# Grid for computing errors and plotting:
x = np.linspace(0, 1, 1000)

# Function f:
s = 5
f = lambda x: np.abs(x - 0.5) ** s

# Loop over L:
Lmax = 12
LL = np.arange(1, Lmax + 1)
e = np.zeros(Lmax)
for L in LL:
    n = ceil(L * log(L**s))  # the size is L * log(1/eps) = L * L^s
    e[L - 1] = np.max(np.abs(f(x) - fhat_L(x, L, s, n)))

plt.figure()
plt.loglog(LL, e, ".-r", label="Numerical error")
plt.loglog(LL, 2 / factorial(s) * 1 / LL**s, "--k", label="Theoretical error")
plt.xlabel("L")
plt.ylabel("Error")
plt.legend()
plt.show()
