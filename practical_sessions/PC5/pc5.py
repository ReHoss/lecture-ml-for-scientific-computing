# ---
# jupyter:
#   jupytext:
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

# %% [markdown]
# <div style="text-align: center;">
# <strong>Machine Learning for Scientific Computing and Numerical Analysis - Practical 5</strong>
# </div>
# <div style="text-align: center;">
# <i>Physics Informed Neural Networks</i>
# </div>
# <div style="text-align: center;">
# </br>
# <p>Victorita Dolean, Loïc Gouarin, Rémy Hosseinkhan, Hadrien Montanelli
# </br>
# 2025-2026
# </p>
# </div>
#
# This notebook combines:
# 1. Automatic differentiation with PyTorch
# 2. PINN setup (networks and grids)
# 3. Elliptic and parabolic PDEs with PINNs

# %% [markdown]
# ---
# # PART 1: Automatic differentiation and PINNs

# %%
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

# %%
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
DTYPE = torch.float32

# %% [markdown]
# # 1 Computing derivatives with PyTorch

# %% [markdown]
# As discussed in Lecture 5, we will use Physics-Informed Neural Networks (PINNs) to solve ODEs and PDEs. These equations involve multiple derivatives with respect to time and space.
#
# To compute these derivatives within our trained model, we rely on automatic differentiation. PyTorch provides automatic differentiation through the `torch.autograd` module.
#
# Let's give a really simple example to explain how it works. Suppose we have the function
#
# $$
# y(x) = x^2,
# $$
#
# and we want to compute $\frac{d y}{d x}$. We can do that with PyTorch with the following code.

# %%
x = torch.tensor(3.0, requires_grad=True)

y = x**2  # y(x) = x^2
y.backward()  # compute dy/dx

x.grad.item()  # y'(3) = 6

# %% [markdown]
# We have the possibility to add various options when working with PyTorch tensors.

# %% [markdown]
# ## 1.1 Gradients accumulation

# %% [markdown]
# If we try to call backward once more, the gradients will accumulate. We need to zero them out first.

# %%
x = torch.tensor(3.0, dtype=DTYPE, requires_grad=True)
y = x**2
y.backward()
print(x.grad)

# Gradients accumulate, so we zero them
x.grad.zero_()
y = x**2
y.backward()
print(x.grad)

# %% [markdown]
# ## 1.2 Computing higher-order derivatives

# %% [markdown]
# To compute higher-order derivatives, we need to set `create_graph=True` when calling `backward()`. This allows us to differentiate through the gradient computation.
#
# Let's take this simple example:

# %%
x = torch.tensor(3.0, dtype=DTYPE, requires_grad=True)

y = x**2
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]

print(dy_dx)

# Now we can compute the second derivative
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]
print(d2y_dx2)

# %% [markdown]
# ## Exercise
#
# Consider the following function
#
# $$
# f(x, y) = \sin(2\pi x)y.
# $$
#
# Compute $\frac{\partial^2 f}{\partial x \partial y}$.

# %%
x = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
y = torch.tensor(5.0, dtype=DTYPE, requires_grad=True)

f = None  # replace me (use torch.sin)
df_dy = None  # replace me
d2f_dxy = None  # replace me

# %% [markdown]
# ### Validation

# %%
d2f_dxy.item()  # you should get 2*pi

# %% [markdown]
# We can use automatic differentiation in PyTorch on almost every PyTorch tensor. The following example plots a sigmoid function and its derivative.

# %%
x = torch.linspace(-10.0, 10.0, 201, dtype=DTYPE, requires_grad=True)

y = torch.sigmoid(x)
y_sum = y.sum()
y_sum.backward()
dy_dx = x.grad

plt.plot(x.detach().numpy(), y.detach().numpy(), label="y")
plt.plot(x.detach().numpy(), dy_dx.numpy(), label="dy/dx")
plt.legend()
_ = plt.xlabel("x")


# %% [markdown]
# If you'd like to learn more about automatic differentiation in PyTorch, check out https://pytorch.org/docs/stable/autograd.html.

# %% [markdown]
# # 2 Setting up PINNs

# %% [markdown]
# # 2.1 Neural network class

# %% [markdown]
# ## Exercise
#
# Complete the missing parts in the creation of the neural network model.


# %%
class NeuralNet(torch.nn.Module):
    """
    Creates a neural network model with the specified parameters.

    Parameters
    ----------
    neurons : list of int
        A list where each element represents the number of neurons in a hidden layer.
    input_size : int, optional
        The size of the input layer. Default is 1.
    output_size : int, optional
        The size of the output layer. Default is 1.
    activation : str, optional
        The activation function to use for the hidden layers. Default is 'tanh'.
    """

    def __init__(self, neurons, input_size=1, output_size=1, activation="tanh"):
        super(NeuralNet, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation

        # Choose activation function
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        else:
            self.activation = torch.tanh

        # Create layers
        self.layers = torch.nn.ModuleList()

        # Input layer
        self.layers.append(None)  # replace me

        # Hidden layers
        for neurons_count in neurons:
            self.layers.append(None)  # replace me

        # Output layer
        self.layers.append(None)  # replace me

    def forward(self, x):
        # Pass through input layer
        x = self.activation(self.layers[0](x))

        # Pass through hidden layers
        for i in range(1, len(self.layers) - 1):
            x = self.activation(self.layers[i](x))

        # Output layer (no activation)
        x = self.layers[-1](x)

        return x


def create_model(neurons, *, input_size=1, output_size=1, activation="tanh"):
    """
    Creates a neural network model with the specified parameters.

    Returns
    -------
    NeuralNet
        A PyTorch neural network model with the specified architecture.
    """
    model = NeuralNet(
        neurons, input_size=input_size, output_size=output_size, activation=activation
    )
    return model


# %% [markdown]
# ## Exercise
#
# Complete the missing parts in the creation of the 1D grid.


# %%
def grid1D(n_r, dom, gridtype):
    """
    Generates a 1D grid of points based on the specified grid type.

    Parameters
    ----------
    n_r : int
        The number of points in the grid.
    dom : tuple of float
        The domain limits (lower bound, upper bound).
    gridtype : str
        The type of grid to generate. Options are:
        - 'rand': Randomly distributed points in the domain.
        - 'uniform': Equally spaced points in the domain.
        - 'midpoint': Midpoints of a uniform grid.

    Returns
    -------
    x_r : torch.Tensor
        The grid points of shape (n_r, 1).
    x_b : torch.Tensor
        The boundary points of shape (2,), corresponding to `dom[0]` and `dom[1]`.

    Notes
    -----
    - The random grid uses `torch.rand` to sample points from a uniform distribution.
    - The uniform grid uses `torch.linspace` to create evenly spaced points.
    - The midpoint grid is computed as the midpoints of a uniform grid.
    """

    if gridtype == "rand":

        x_r = None  # replace me (use torch.rand)

    elif gridtype == "uniform":

        x_r = None  # replace me (use torch.linspace)

    elif gridtype == "midpoint":

        x_r = torch.linspace(dom[0], dom[1], n_r + 1)
        x_r = 1 / 2 * (x_r[0:n_r] + x_r[1 : n_r + 1])
        x_r = x_r.reshape(n_r, 1)

    x_b = torch.tensor([dom[0], dom[1]])

    return x_r, x_b


# %% [markdown]
# ## Exercise
#
# Complete the missing parts in the creation of the 2D grid.


# %%
def grid2D(n_r, n_b, n_i, dom):
    """
    Generates a 2D grid of points for solving PDEs on a rectangular domain.

    Parameters
    ----------
    n_r : int
        The number of residual points inside the domain.
    n_b : int
        The number of boundary points along each spatial boundary.
    n_i : int
        The number of initial condition points.
    dom : tuple of float
        The domain limits in the form (t_min, t_max, x_min, x_max).

    Returns
    -------
    X_r : torch.Tensor
        The residual points of shape (n_r, 2), representing (t, x) pairs.
    X_b : torch.Tensor
        The boundary points of shape (n_i + 2*n_b, 2), representing (t, x) pairs.
    u_b : torch.Tensor
        The boundary condition values at the boundary points of shape (n_i + 2*n_b, 1).

    Notes
    -----
    - The domain is defined as `[t_min, t_max] x [x_min, x_max]`.
    - Residual points `(t_r, x_r)` are sampled uniformly from the domain.
    - Boundary points include:
        - Initial condition = left boundary (`t_min`, varying `x`).
        - Boundary conditions = top and bottom boudaries (`x_min`, varying `t` and `x_max`, varying `t`).
    - The function `u0(x)` provides the initial condition values at `t_min`.
    """

    # PDE is on [t_min, t_max] x [x_min, x_max]:
    t_min = dom[0]
    t_max = dom[1]
    x_min = dom[2]
    x_max = dom[3]

    # Residual points in time:
    t_r = None  # replace me (use torch.rand)

    # Residual points in space:
    x_r = None  # replace me (use torch.rand)

    # Residual points in time and space:
    X_r = torch.cat([t_r, x_r], dim=1)

    # Boundary points in time:
    t_left = None  # replace me (use torch.rand)
    t_top = None  # replace me (use torch.rand)
    t_bottom = None  # replace me (use torch.rand)
    t_b = torch.cat([t_left, t_top, t_bottom], dim=0)

    # Boundary points in space:
    x_left = None  # replace me (use torch.rand)
    x_top = None  # replace me (use torch.rand)
    x_bottom = None  # replace me (use torch.rand)
    x_b = torch.cat([x_left, x_top, x_bottom], dim=0)

    # Boundary points in time and space:
    X_b = torch.stack([t_b, x_b], dim=1)

    # Boundary data:
    u_left = u0(x_left)
    u_top = 0 * t_top
    u_bottom = 0 * t_bottom
    u_b = torch.cat([u_left, u_top, u_bottom], dim=0)
    u_b = u_b.reshape(n_i + 2 * n_b, 1)

    return X_r, X_b, u_b


# %% [markdown]
# # PART 2: Solving elliptic PDEs with PINNs

# %% [markdown]
# # 3 Solving elliptic PDEs

# %% [markdown]
# ## 3.1 Poisson equation

# %% [markdown]
# We consider the one-dimensional Poisson equation with homogeneous Dirichlet boundary conditions:
#
# \begin{align}
# & -u''(x) = f(x), \quad x \in(-1,1), \\
# & u(-1) = u(1) = 0.
# \end{align}
#
# We will take the following right-hand side and exact solution,
#
# $$
# f(x) = (2\pi)^2\sin(2\pi x), \quad u(x) = \sin(2\pi x).
# $$

# %% [markdown]
# ## Exercise
#
# Complete the missing parts in the computation of the residual, loss, and gradient. Then:
#
# - setup a neural network model with 5 layers with 25 neurons each [Step 1];
# - choose $\gamma=1$ and $n_r=100$ residual points [Step 2];
# - try the three different 1D grids of `grid1D` [Step 2];
# - choose 5000 iterations maximum (`itr_max`) and a stopping criteria of $10^{-3}$ for the loss (`loss_min`) [Step 3].
#
# The displayed $L^\infty$-error should be around $10^{-3}$.

# %%
###########################################################
# Residual, loss, and gradient
###########################################################

# ODE setup:
dom = [-1, 1]
f = lambda x: (2 * np.pi) ** 2 * torch.sin(2 * np.pi * x)
u_exact = lambda x: np.sin(2 * np.pi * x)


# Define residual of the ODE:
def residual(x, u_xx):
    return None  # replace me


# Residual computation:
def compute_res(model, x_r):

    x_r.requires_grad_(True)
    u = model(x_r)
    u_x = None  # replace me (use torch.autograd.grad)
    u_xx = None  # replace me (use torch.autograd.grad)
    res = model.residual(x_r, u_xx)

    return res


# Loss function computation:
def compute_loss(model, x_r, x_b, u_b, gamma):

    # Loss on residual:
    res = compute_res(model, x_r)
    loss_r = torch.mean(torch.square(res))

    # Loss on boundary points:
    u_pred = model(x_b)
    loss_b = torch.mean(torch.square(u_pred - u_b))

    # Total loss:
    loss = None  # replace me

    return loss, loss_r, loss_b


# Gradient computation:
def compute_grad(model, x_r, x_b, u_b, gamma=1):

    loss, loss_r, loss_b = compute_loss(model, x_r, x_b, u_b, gamma)

    loss.backward()

    return loss, loss_r, loss_b


###########################################################
# Algorithm
###########################################################

# Initialize neural network model (Step 1):
model = None  # replace me (use create_model)
model.residual = residual

# Method parameters (Step 2):
gamma = None  # replace me
n_r = None  # replace me

# Residual and boundary points, and boundary condition (Step 2):
x_r, x_b = None  # replace me (use grid1D)
u_b = torch.tensor([0, 0]).reshape(2, 1)

# Optimization setup (Step 3):
itr_max = None  # replace me
loss_min = None  # replace me
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


def train_step():
    optimizer.zero_grad()
    loss, loss_r, loss_b = compute_grad(model, x_r, x_b, u_b, gamma)
    optimizer.step()

    return loss, loss_r, loss_b


# %% [markdown]
# ### Validation

# %%
itr_max = 5000
loss_min = 1e-3

# Optimization loop:
hist = []
hist_r = []
hist_b = []
loss = 1
itr = 1
t0 = time.time()
while itr < itr_max + 1 and loss > loss_min:
    loss, loss_r, loss_b = train_step()
    hist.append(loss.item())
    hist_r.append(loss_r.item())
    hist_b.append(loss_b.item())
    itr += 1
t1 = time.time()
print(f"Time: {t1-t0:.5f}s")

# Compute neural net and exact solutions on a different grid:
n_plot = 1000
x_plot = np.linspace(dom[0], dom[1], n_plot)
x_plot_tensor = torch.tensor(x_plot, dtype=DTYPE).reshape(-1, 1)
with torch.no_grad():
    u_pred = model(x_plot_tensor).squeeze().numpy()
u_ex = u_exact(x_plot)
error = np.max(np.abs(u_ex - u_pred)) / np.max(np.abs(u_ex))
print(f"Error (L-inf): {error:.2e}")

# Plot loss function:
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].semilogy(range(len(hist)), hist, "k-", label="total loss")
axs[0].semilogy(range(len(hist)), hist_r, "--", label="residual loss")
axs[0].semilogy(range(len(hist)), hist_b, "--", label="boundary loss")
axs[0].set_xlabel("Iteration")
axs[0].set_ylabel("Loss function")
axs[0].set_title("Optimization history")
axs[0].set_xlim(0, itr)
axs[0].grid(1)
axs[0].legend(loc="best", fontsize=10)

# Plot neural net and exact solutions:
axs[1].plot(x_plot, u_pred, label="neural net solution", color="tab:blue")
axs[1].plot(x_plot, u_ex, "--", label="exact solution", color="tab:red")
axs[1].set_xlim(dom[0], dom[1])
axs[1].set_ylim(-2, 2)
axs[1].grid(1)
axs[1].legend(loc="best", fontsize=10)

# %% [markdown]
# ## 3.2 Constant-coefficient elliptic problem

# %% [markdown]
# Let $\mu>0$. We consider the following one-dimensional elliptic problem with homogeneous Dirichlet boundary conditions:
#
# \begin{align}
# & -u''(x) + \mu u(x) = f(x), \quad x \in(-1,1), \\
# & u(-1) = u(1) = 0.
# \end{align}
#
# We will take the following right-hand side and exact solution for $\mu=1$,
#
# $$
# f(x) = [(2\pi)^2 + \mu]\sin(2\pi x), \quad u(x) = \sin(2\pi x).
# $$

# %% [markdown]
# ## Exercise
#
# Complete the missing parts in the computation of the residual. Then:
#
# - setup a neural network model with 5 layers with 25 neurons each [Step 1];
# - choose $\gamma=1$ and $n_r=100$ residual points [Step 2];
# - try the three different 1D grids of `grid1D` [Step 2];
# - choose 5000 iterations maximum (`itr_max`) and a stopping criteria of $10^{-3}$ for the loss (`loss_min`) [Step 3].
#
# The displayed $L^\infty$-error should be around $10^{-3}$.

# %%
###########################################################
# Residual
###########################################################

# ODE setup:
dom = [-1, 1]
mu = 1
f = lambda x: ((2 * np.pi) ** 2 + mu) * torch.sin(2 * np.pi * x)
u_exact = lambda x: np.sin(2 * np.pi * x)


# Define residual of the ODE:
def residual(x, u, u_xx):
    return None  # replace me


# Residual computation:
def compute_res(model, X_r):

    X_r.requires_grad_(True)
    u = model(X_r)
    u_x = None  # replace me (use torch.autograd.grad)
    u_xx = None  # replace me (use torch.autograd.grad)
    res = model.residual(X_r, u, u_xx)

    return res


###########################################################
# Algorithm
###########################################################

# Initialize neural network model (Step 1):
model = None  # replace me (use create_model)
model.residual = residual

# Method parameters (Step 2):
gamma = None  # replace me
n_r = None  # replace me

# Residual and boundary points, and boundary condition (Step 2):
x_r, x_b = None  # replace me (use grid1D)
u_b = torch.tensor([0, 0]).reshape(2, 1)

# Optimization setup (Step 3):
itr_max = None  # replace me
loss_min = None  # replace me
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


def train_step():
    optimizer.zero_grad()
    loss, loss_r, loss_b = compute_grad(model, x_r, x_b, u_b, gamma)
    optimizer.step()

    return loss, loss_r, loss_b


# %% [markdown]
# ### Validation

# %%
itr_max = 5000
loss_min = 1e-3

# Optimization loop:
hist = []
hist_r = []
hist_b = []
loss = 1
itr = 1
t0 = time.time()
while itr < itr_max + 1 and loss > loss_min:
    loss, loss_r, loss_b = train_step()
    hist.append(loss.item())
    hist_r.append(loss_r.item())
    hist_b.append(loss_b.item())
    itr += 1
t1 = time.time()
print(f"Time: {t1-t0:.5f}s")

# Compute neural net and exact solutions on a different grid:
n_plot = 1000
x_plot = np.linspace(dom[0], dom[1], n_plot)
x_plot_tensor = torch.tensor(x_plot, dtype=DTYPE).reshape(-1, 1)
with torch.no_grad():
    u_pred = model(x_plot_tensor).squeeze().numpy()
u_ex = u_exact(x_plot)
error = np.max(np.abs(u_ex - u_pred)) / np.max(np.abs(u_ex))
print(f"Error (L-inf): {error:.2e}")

# Plot loss function:
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].semilogy(range(len(hist)), hist, "k-", label="total loss")
axs[0].semilogy(range(len(hist)), hist_r, "--", label="residual loss")
axs[0].semilogy(range(len(hist)), hist_b, "--", label="boundary loss")
axs[0].set_xlabel("Iteration")
axs[0].set_ylabel("Loss function")
axs[0].set_title("Optimization history")
axs[0].set_xlim(0, itr)
axs[0].grid(1)
axs[0].legend(loc="best", fontsize=10)

# Plot neural net and exact solutions:
axs[1].plot(x_plot, u_pred, label="neural net solution", color="tab:blue")
axs[1].plot(x_plot, u_ex, "--", label="exact solution", color="tab:red")
axs[1].set_xlim(dom[0], dom[1])
axs[1].set_ylim(-2, 2)
axs[1].grid(1)
axs[1].legend(loc="best", fontsize=10)


# %% [markdown]
# ---
# # PART 3: Solving parabolic PDEs with PINNs

# %% [markdown]
# # 4 Solving parabolic PDEs

# %% [markdown]
# ## 4.1 Heat equation

# %% [markdown]
# We consider the one-dimensional heat equation with homogeneous Dirichlet boundary conditions:
#
# \begin{align}
# & u_t(t,x) = u_{xx}(x,t), && t\in[0,T], \quad x\in(-1,1), \\
# & u(t,x) = 0, && t\in[0,T], \\
# & u(0,x) = u_0(x), && x\in(-1,1).
# \end{align}
#
# We will take $T=10^{-2}$, and the following initial condition and exact solution,
#
# $$
# u_0(x) = \sin(\pi x), \quad u(t,x) = \sin(\pi x)e^{-\pi^2 t}.
# $$

# %% [markdown]
# ## Exercise
#
# Complete the missing parts in the computation of the residual, loss, and gradient. Then:
#
# - setup a neural network model with 5 layers with 25 neurons each [Step 1];
# - choose $\gamma=10$, $n_r=200$ residual points, and $n_b=n_i=20$ boundary points [Step 2];
# - setup the grid with `grid2D` [Step 2];
# - choose 5000 iterations maximum (`itr_max`) and a stopping criteria of $10^{-5}$ for the loss (`loss_min`) [Step 3].
#
# The displayed $L^\infty$-error should be around $10^{-2}$.

# %%
###########################################################
# Residual, loss, and gradient
###########################################################

# PDE setup:
T = 1e-2
dom = [0, T, -1, 1]
u0 = lambda x: torch.sin(np.pi * x)
u_exact = lambda t, x: np.sin(np.pi * x) * np.exp(-((np.pi) ** 2) * t)


# Define residual of the PDE:
def residual(model, x, u, u_t, u_xx):
    return None  # replace me


# Residual computation:
def compute_res(model, X_r):

    t_r = X_r[:, 0:1]
    x_r = X_r[:, 1:2]

    t_r.requires_grad_(True)
    x_r.requires_grad_(True)
    g = torch.cat([t_r, x_r], dim=1)
    u = model(g)
    u_t = None  # replace me (use torch.autograd.grad)
    u_x = None  # replace me (use torch.autograd.grad)
    u_xx = None  # replace me (use torch.autograd.grad)
    res = model.residual(model, X_r, u, u_t, u_xx)

    return res


# Loss function computation:
def compute_loss(model, X_r, X_b, u_b, gamma):

    # Loss on residual:
    res = compute_res(model, X_r)
    loss_r = torch.mean(torch.square(res))

    # Loss on boundary points:
    u_pred = model(X_b)
    loss_b = torch.mean(torch.square(u_pred - u_b))

    # Total loss:
    loss = None  # replace me

    return loss, loss_r, loss_b


# Gradient computation:
def compute_grad(model, X_r, X_b, u_b, gamma=1):

    loss, loss_r, loss_b = compute_loss(model, X_r, X_b, u_b, gamma)
    loss.backward()

    return loss, loss_r, loss_b


###########################################################
# Algorithm
###########################################################

# Initialize neural network model (Step 1):
model = None  # replace me
model.residual = residual

# Method parameters (Step 2):
gamma = None  # replace me
n_r = None  # replace me
n_b = None  # replace me
n_i = None  # replace me

# Residual and boundary points, and boundary condition (Step 2):
X_r, X_b, u_b = None  # replace me (use grid 2D)

# Optimization setup (Step 3):
itr_max = None  # replace me
loss_min = None  # replace me
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


def train_step():
    optimizer.zero_grad()
    loss, loss_r, loss_b = compute_grad(model, X_r, X_b, u_b, gamma)
    optimizer.step()

    return loss, loss_r, loss_b


# %% [markdown]
# ### Validation

# %%
# Optimization loop:
hist = []
hist_r = []
hist_b = []
loss = 1
itr = 1
t0 = time.time()
while itr < itr_max + 1 and loss > loss_min:
    loss, loss_r, loss_b = train_step()
    hist.append(loss.item())
    hist_r.append(loss_r.item())
    hist_b.append(loss_b.item())
    itr += 1
t1 = time.time()
print(f"Time: {t1-t0:.5f}s")

# Compute neural net and exact solutions on a different grid:
n_plot = 100
x_plot = np.linspace(-1, 1, n_plot)
u_pred = np.zeros(n_plot)
for i in range(n_plot):
    tmp = torch.tensor([T, x_plot[i]]).reshape(1, 2)
    with torch.no_grad():
        u_pred[i] = model(tmp).item()
u_ex = u_exact(T, x_plot)
error = np.max(np.abs(u_ex - u_pred)) / np.max(np.abs(u_ex))
print(f"Error (L-inf): {error:.2e}")

# Plot loss function:
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].semilogy(range(len(hist)), hist, "k-", label="total loss")
axs[0].semilogy(range(len(hist)), hist_r, "--", label="residual loss")
axs[0].semilogy(range(len(hist)), hist_b, "--", label="boundary loss")
axs[0].set_xlabel("Iteration")
axs[0].set_ylabel("Loss function")
axs[0].set_title("Optimization history")
axs[0].set_xlim(0, itr)
axs[0].grid(1)
axs[0].legend(loc="best", fontsize=10)

# Plot neural net and exact solutions:
axs[1].plot(x_plot, u_pred, label="neural net solution", color="tab:blue")
axs[1].plot(x_plot, u_ex, "--", label="exact solution", color="tab:red")
axs[1].set_xlim(dom[2], dom[3])
axs[1].set_ylim(-2, 2)
axs[1].grid(1)
axs[1].legend(loc="best", fontsize=10)

# %% [markdown]
# ## 4.2 Allen-Cahn equation

# %% [markdown]
# We consider the one-dimensional Allen-Cahn equation with homogeneous Dirichlet boundary conditions:
#
# \begin{align}
# & u_t(t,x) = \epsilon u_{xx}(x,t) + u - u^3, && t\in[0,T], \quad x\in(-1,1), \\
# & u(t,x) = 0, && t\in[0,T], \\
# & u(0,x) = u_0(x), && x\in(-1,1).
# \end{align}
#
# We will take $T=1$, $\epsilon=10^{-2}$, and the following initial condition,
#
# $$
# u_0(x) = e^{-100(x+0.5)^2} - e^{-100(x-0.5)^2}.
# $$

# %% [markdown]
# ## Exercise
#
# Complete the missing parts in the computation of the residual. Then:
#
# - setup a neural network model with 5 layers with 25 neurons each [Step 1];
# - choose $\gamma=10$, $n_r=500$ residual points, and $n_b=n_i=50$ boundary points [Step 2];
# - setup the grid with `grid2D` [Step 2];
# - choose 5000 iterations maximum (`itr_max`) and a stopping criteria of $10^{-5}$ for the loss (`loss_min`) [Step 3].
#
# The displayed $L^\infty$-error should be around $10^{-2}$.

# %%
###########################################################
# Residual
###########################################################

# PDE setup:
T = 1
dom = [0, T, -1, 1]
eps = 1e-2
u0 = lambda x: torch.exp(-100 * (x + 0.5) ** 2) - torch.exp(-100 * (x - 0.5) ** 2)
u_ex = np.loadtxt("uex-ac.txt")


# Define residual of the PDE:
def residual(model, x, u, u_t, u_xx):
    return None  # replace me


###########################################################
# Algorithm
###########################################################

# Initialize neural network model (Step 1):
model = create_model([25] * 5, input_size=2)
model.residual = residual

# Method parameters (Step 2):
gamma = None  # replace me
n_r = None  # replace me
n_b = None  # replace me
n_i = None  # replace me

# Residual and boundary points, and boundary condition (Step 2):
X_r, X_b, u_b = None  # replace me (use grid 2D)

# Optimization setup (Step 3):
itr_max = None  # replace me
loss_min = None  # replace me
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


def train_step():
    optimizer.zero_grad()
    loss, loss_r, loss_b = compute_grad(model, X_r, X_b, u_b, gamma)
    optimizer.step()

    return loss, loss_r, loss_b


# %% [markdown]
# ### Validation

# %%
# Optimization loop:
hist = []
hist_r = []
hist_b = []
loss = 1
itr = 1
t0 = time.time()
while itr < itr_max + 1 and loss > loss_min:
    loss, loss_r, loss_b = train_step()
    hist.append(loss.item())
    hist_r.append(loss_r.item())
    hist_b.append(loss_b.item())
    itr += 1
t1 = time.time()
print(f"Time: {t1-t0:.5f}s")

# Compute neural net and exact solutions on a different grid:
n_plot = 100
x_plot = np.linspace(-1, 1, n_plot)
u_pred = np.zeros(n_plot)
for i in range(n_plot):
    tmp = torch.tensor([T, x_plot[i]]).reshape(1, 2)
    with torch.no_grad():
        u_pred[i] = model(tmp).item()
error = np.max(np.abs(u_ex - u_pred)) / np.max(np.abs(u_ex))
print(f"Error (L-inf): {error:.2e}")

# Plot loss function:
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].semilogy(range(len(hist)), hist, "k-", label="total loss")
axs[0].semilogy(range(len(hist)), hist_r, "--", label="residual loss")
axs[0].semilogy(range(len(hist)), hist_b, "--", label="boundary loss")
axs[0].set_xlabel("Iteration")
axs[0].set_ylabel("Loss function")
axs[0].set_title("Optimization history")
axs[0].set_xlim(0, itr)
axs[0].grid(1)
axs[0].legend(loc="best", fontsize=10)

# Plot neural net and exact solutions:
axs[1].plot(x_plot, u_pred, label="neural net solution", color="tab:blue")
axs[1].plot(x_plot, u_ex, "--", label="exact solution", color="tab:red")
axs[1].set_xlim(dom[2], dom[3])
axs[1].set_ylim(-2, 2)
axs[1].grid(1)
axs[1].legend(loc="best", fontsize=10)
