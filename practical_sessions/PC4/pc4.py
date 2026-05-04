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
# <strong>Machine Learning for Scientific Computing and Numerical Analysis - Practical 4</strong>
# </div>
# <div style="text-align: center;">
# <i>Finite Element Method</i>
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
# 1. Approximation of a function (P1 elements)
# 2. Elliptic problems (Poisson)
# 3. Time-dependent problems (Heat, Allen-Cahn)

# %%
from math import pi
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import eigvals, expm
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
import time
try:
    from helpers import elliptic_reference_solution, heat_eq_exact, plot_series
except ImportError:
    pass

np.random.seed(42)

# %% [markdown]
# ---
# # PART 1: Finite element method for function approximation

# %% [markdown]
# # 1 Approximation of a function by a P1 finite element basis

# %% [markdown]
# In this notebook we recall a few basic elements about the P1 approximation of functions.


# %% [markdown]
# ## 1.1 Shape function

# %% [markdown]
# Our goal is to approximate a function $u: [-1, 1] \to \mathbb{R}$ with a P1 finite element basis.

# %% [markdown]
# First we define a shape function (sometimes called "hat function")
# $\phi: \mathbb{R} \to \mathbb{R}$:
# $$
# \phi(x) = \left\{
# \begin{array}{ll}
# 1 - |x| & \text{if } |x| \leq 1, \\
# 0 & \text{otherwise}.
# \end{array}
# \right.
# $$

# %% [markdown]
# ## Exercise
#
# Define the function $\phi$ using a lambda function.

# %%
# Shape function:
phi = lambda x: "To implement"

# %% [markdown]
# ### Validation

# %%
# Plot shape function:
xeval = np.linspace(-1, 1, 1000) # evaluation grid
plt.figure()
plt.plot(xeval, phi(xeval), '*', label=r'shape function $\phi$')
plt.xlabel(r"$x$")
plt.legend()
plt.grid()


# %% [markdown]
# ## 1.2 Basis functions

# %% [markdown]
# Consider now the set of $m+2$ points $x_j = -1 + jh$, $h = 2/(m+1)$, for $j=0,\ldots,m+1$, which yields a discretization grid of $[-1,1]$.


# %%
# Grid:
m = 100
h = 2/(m + 1)
xx = np.linspace(-1, 1, m + 2)


# %% [markdown]
# The function $\phi$ allows us to define a basis of functions $\Phi_j$:
# $$
# \Phi_j(x) = \phi\left(\frac{x - x_j}{h}\right),
# \qquad
# j=1,\ldots,m.
# $$


# %% [markdown]
# ## Exercise
#
# Define the function $\Phi_j(x) = \Phi(j,x)$ using a two-parameter lambda function.
# %%
Phi = lambda j, x: "To implement"

# %% [markdown]
# ### Validation

# %%
# Plot basis functions:
m = 10
meval = 1000
h = 2.0/(m + 1)
heval = 2.0/meval
xx = np.linspace(-1, 1, m + 2) # computation grid
xeval = np.linspace(-1, 1, meval + 1) # evaluation grid
plt.figure()
plt.grid()
plt.xlabel(r"$x$")
for j in range(1, m + 1):
    plt.plot(xeval, Phi(j, xeval), 'k')

# %% [markdown]
# ## 1.3 Approximating a function

# %% [markdown]
# Now we want to approximate a function $u$. Suppose that $u(x=\pm 1) = 0$.

# %%
# An example of function u:
u = lambda x: np.sin(np.pi*x)*np.cos(-x**2+5)

# %% [markdown]
# We can approximate $u$ via a linear combination $u_h$ of the basis functions $\Phi_j$:
# $$
# u_h(x) = \sum_{j=1}^m u_j \Phi_j(x),
# \qquad
# u_j = u(x_j),
# \qquad
# j=1,\ldots,m.
# $$
# %% [markdown]
# ## Exercise
#
# Define the function $u_h$ using a one parameter lambda function.
# %%
uh = lambda x: "To implement"
# %% [markdown]
# ### Validation

# %%
plt.figure()
plt.grid()
plt.xlabel(r"$x$")
plt.plot(xeval, uh(xeval), '*')
plt.plot(xeval, u(xeval), '-');
# %% [markdown]
# ## 1.4 Error and convergence

# %% [markdown]
# ## Exercise
#
# Compute the error $e_h$ for $m=1,\ldots,500$.
# %%
m_values = np.arange(1, 500)
eh_values = np.zeros_like(m_values, dtype=float)
for i,m in enumerate(m_values):
    h = 2.0/(m + 1)
    xx = None # To implement
    Phi = None # To implement
    uh = None # To implement
    eh_values[i] = None #"To implement"


# %% [markdown]
# ### Validation
# %%
plt.figure()
plt.grid()
plt.xlabel(r"$m$")
plt.ylabel(r"$e_h$")
plt.loglog()
plt.plot(m_values, eh_values, '*');


# %% [markdown]
# ---
# # PART 2: Finite element method for elliptic problems
# # 2 Poisson equation with homogeneous Dirichlet boundary conditions

# %% [markdown]
# We consider the one-dimensional Poisson equation with homogeneous Dirichlet boundary conditions:
#
# \begin{align}
# -u''(x) &= f(x) && \text{in } \Omega = (-1,1),\\
# u(-1) &= u(1) = 0.
# \end{align}
#
# The variational formulation of the problem is to find $u\in V=\{v\in H^1(-1,1),\,v(\pm1)=0\}$ such that
#
# $$
# \int_{\Omega} u'(x)v'(x)dx = \int_{\Omega} f(x)v(x)dx,
# $$
#
# for all $v\in V$.
#
# We discretize the interval $[-1,1]$ using a set of $m+2$ points 
# $$
# x_j = -1 + jh, \qquad h = 2/(m+1),\qquad j=0,\ldots,m+1,
# $$ 
# which provides associate basis functions $x\mapsto\phi_j(x)$, $j=1,\ldots,m$.
#
# We seek a finite element approximation of $u$
#
# $$
# u_h(x)
# = \sum_{j=1}^m u_j\phi_j(x)
# $$
# that verifies the Galerkin variational problem
#
# $$
# \int_{\Omega} u_h'(x)\phi_i'(x)  dx  
# =
# \int_{\Omega} f(x)\phi_i(x) dx
# \quad \text{for all $i=1,\ldots,m$}.
# $$
#
# By substituting the expression of $u_h$ into the weak formulation, we obtain the following linear system
#
# $$
# \sum_{j=1}^{m} u_j
# \int_{\Omega} \phi_i'(x)\phi_j'(x)dx
# =
# \int_{\Omega} f(x)\phi_i(x) dx
# \quad \text{for all $i=1,\ldots,m$.}
# $$ 
#
# We set
# $$
# A_{ij} = \int_{\Omega} \phi_i'(x)\phi_j'(x)dx,
# \quad \text{and} \quad
# F_i = \int_{\Omega} f(x)\phi_i(x)dx.
# $$
# Then the linear system can be written as
# $$
# A U = F,
# $$
# where 
# $$
# U = (u_1,\ldots,u_m)^T,
# \quad F = (F_1,\ldots,F_m)^T,
# \quad A= (A_{ij})_{1\leq i,j \leq m}
# .
# $$
#
# Let $T$ be the $m\times m$ tridiagonal matrix defined by $T_{i,i+1} = T_{i+1,i} = 1$ for $i=1,\ldots,m-1$ and $T_{i,j} = 0$ otherwise:
#
# $$
# T =
# \begin{pmatrix}
# 0 & 1 & 0 & \cdots & 0\\
# 1 & 0 & 1 & \ddots & \vdots\\
# 0 & \ddots & \ddots & \ddots & 0\\
# \vdots & \ddots & 1 & 0 & 1\\
# 0 & \cdots & 0 & 1 & 0\\
# \end{pmatrix}.
# $$
#
# Then the matrices A can be expressed as
# $$
# A =\frac{1}{h}
# \left(
# 2 I - T
# \right)
# .
# $$
#
# In order to solve the linear system $AU = F$, we need to evaluate the right-hand side $F$. We propose to
# use the following approximation
# $$
# F_i = 
# \int_{\Omega} f(x) \phi_i(x) dx \approx
# \int_{\Omega} \sum_{j=1}^{m} f(x_j) \phi_j(x)\phi_i(x) dx,
# $$
# which also reads
# $$
# F = M \begin{pmatrix} f(x_1) \\ \vdots \\ f(x_m) \end{pmatrix}
# ,\quad
# M_{ij} = \int_{\Omega} \phi_j(x)\phi_i(x) dx,
# \quad
# M = (M_{ij})_{1\leq i,j \leq m}.
# $$
#
# The matrix $M$ can also be expressed thanks to $T$
#
# $$
# M =
# \frac{h}{3}
# \left(
# 2I + \frac{1}{2}T
# \right)
# .
# $$

# %% [markdown]
# ## 2.1 Assembling matrices

# %% [markdown]
# ## Exercise
#
# Assemble the matrix A and M.
# %%
def assemble_matrices(m, h):
    # assemble the matrices A and M
    # hints: use the diag() and eye() functions from scipy.sparse 
    return A, M

# %% [markdown]
# ### Validation

# %%
m = 10
h = 2/(m + 1)
A, M = assemble_matrices(m, h)
print(A.todense()) # the first line reads : [11.  -5.5  0.   0.   0.   0.   0.   0.   0.   0. ]
print(M.todense()) # the first line reads: [0.12121212 0.03030303 0.0 ... 0.0]


# %% [markdown]
# ## 2.2 Assembling the rhs

# %% [markdown]
# ## Exercise
#
# Assemble the right-hand side of the linear system.

# %%
def poisson_RHS(M, f, xx):
    return None # To implement
# %% [markdown]
# ### Validation

# %%
nu = 1
f = lambda x: (2*pi*nu)**2*np.sin(2*pi*nu*x)

m = 10
h = 2/(m + 1)
xx = np.linspace(-1, 1, m + 2)
A, M = assemble_matrices(m, h)

RHS = poisson_RHS(M, f, xx)

print(RHS) # the first resulting parameters should be  5.25694316  4.36762623 -1.62818815

# %% [markdown]
# ## 2.3 Solving the linear system

# %% [markdown]
# The final setp is to solve $AU = F$.

# %% [markdown]
# ## Exercise
#
# Assemble the matrices, the rhs, and solve the linear system in the following routine.

# %%
def solve_poisson(m, f):
    h = 2/(m + 1)
    xx = np.linspace(-1, 1, m + 2) # computation grid

    start = time.time() 
    A, M = None, None # To implement: assemble matrices
    F = None # To implement: assemble RHS
    end = time.time()
    print(f'Time  (setup): {end-start:.5f}s')
 
    start = time.time() 
    U = None # To implement: solve linear system
    end = time.time()
    print(f'Time  (solve): {end-start:.5f}s')
    return U


# %% [markdown]
# ### Validation

# %%
# Setup (problem):
nu = 1
f = lambda x: (2*pi*nu)**2*np.sin(2*pi*nu*x)

# Setup (grid):
xeval = np.linspace(-1, 1, 1000) # evaluation grid
m = 100
h = 2/(m + 1)
xx = np.linspace(-1, 1, m + 2) # computation grid

# Solve Poisson equation:
U = solve_poisson(m, f)

# Evalute the approximate solution on the evaluation grid:
I, XEVAL = np.meshgrid(np.arange(1, m + 1), xeval)
u = Phi(I, XEVAL) @ U

# Compute exact solution:
uex = lambda x: np.sin(2*pi*nu*x)

# Compute the L-inf error:
error = np.max(np.abs(uex(xeval) - u))/np.max(np.abs(uex(xeval)))
print(f'Error (L-inf): {error:.2e}')

# Plot:
plt.figure()
plt.grid()
plt.xlabel(r"$x$")
plt.plot(xeval, u, '*',label=r'$u_h$')
plt.plot(xeval, uex(xeval), '-',label=r'$u_\mathrm{exact}$');
plt.legend();


# %% [markdown]
# # 3 Constant-coefficient elliptic problem with homogeneous Dirichlet boundary conditions

# %% [markdown]
# Let $\mu>0$. We consider the following one-dimensional ellipic problem with homogeneous Dirichlet boundary conditions.
# \begin{align}
# -u''(x) + \mu u(x) &= f(x) && \text{in } \Omega = (-1,1),\\
# u(-1) &= u(1) = 0.
# \end{align}
#
# The variational formulation of the problem is to find $u\in V$ such that
# $$
# \int_{\Omega} u'(x)v'(x)dx + \int_{\Omega} \mu u(x)v(x)dx = \int_{\Omega} f(x)v(x)dx
# $$
# for all $v\in V$.
#
# We seek a finite element approximation of $u$ of the form
#
# $$
# u_h(x)
# = \sum_{j=1}^m u_j\phi_j(x),
# $$
# which verifies the Galerkin variational problem
#
# $$
# \int_{\Omega} u_h'(x)\phi_i'(x)  dx + \int_{\Omega} \mu u_h(x) \phi_i(x)dx  
# =
# \int_{\Omega} f(x)\phi_i(x) dx
# \quad \text{for all $i=1,\ldots,m$.}
# $$
#
# By substituting the expression of $u_h$ into the weak formulation, we obtain the following linear system
#
# $$
# \sum_{j=1}^m u_j
# \left(
# \int_{\Omega} \phi_i'(x)\phi_j'(x)dx
# +
# \int_{\Omega} \mu \phi_i(x)\phi_j(x)dx
# \right) 
# =
# \int_{\Omega} f(x) \phi_j(x)dx
# \quad \text{for all $i=1,\ldots,m$.}
# $$ 
#
# Using the matrices $A$ and $M$ above, we obtain
# $$
# (A + \mu M) U = F,
# $$
# where have set once again
# $$
# U = (u_1,\ldots,u_m)^T,
# \quad F = (F_1,\ldots,F_m)^T,
# $$
# with the same approximation for $F$, i.e.,
# $
# F = M \begin{pmatrix} f(x_1) \\ \vdots \\ f(x_m) \end{pmatrix}.
# $

# %% [markdown]
# ## Exercise
#
# Implement the solver for this problem.

# %%
def solve_elliptic(m, f, mu):
    h = 2/(m + 1)
    xx = np.linspace(-1, 1, m + 2) # computation grid

    start = time.time() 
    A, M = None, None # To implement: assemble matrices
    F = None # To implement: assemble RHS
    end = time.time()
    print(f'Time  (setup): {end-start:.5f}s')
 
    start = time.time() 
    U = None # To implement: solve linear system
    end = time.time()
    print(f'Time  (solve): {end-start:.5f}s')
    
    return U


# %% [markdown]
# ### Validation

# %%
# Setup (problem):
mu = 1.3
nu = 1
B = 1.8
f = lambda x: B*((2*pi*nu)**2 + mu)*np.sin(2*pi*nu*x)

# Setup (grid):
xeval = np.linspace(-1, 1, 1000) # evaluation grid
m = 100
h = 2/(m + 1)
xx = np.linspace(-1, 1, m + 2) # computation grid

# Solve the elliptic problem:
U = solve_elliptic(m, f, mu)

# Evalute the approximate solution on the evaluation grid:
I, XEVAL = np.meshgrid(np.arange(1, m + 1), xeval)
u = Phi(I, XEVAL) @ U

# Compute exact solution:
uex = lambda x: B*np.sin(2*pi*nu*x) # exact solution

# Compute error:
error = np.max(np.abs(uex(xeval) - u))/np.max(np.abs(uex(xeval)))
print(f'Error (L-inf): {error:.2e}')

# Plot:
plt.figure()
plt.grid()
plt.xlabel(r"$x$")
plt.plot(xeval, u, '*',label=r'$u_h$')
plt.plot(xeval, uex(xeval), '-',label=r'$u_\mathrm{exact}$');
plt.legend();

# %% [markdown]
# ---
# # PART 3: Finite element approximation for time-dependant problems
#
# # 2 Heat equation with homogeneous Dirichlet boundary conditions

# %% [markdown]
# In this section we are going to study a discretization of the one-dimensional heat equation
# using the finite element method.
#
# We note $\Omega=(-1,1)$ and $V=\{v\in H^1(-1,1),\,v(\pm1)=0\}$.
#
# We consider the one-dimensional heat equation with homogeneous Dirichlet boundary conditions.
#
# \begin{align}
# u_t(x,t) &= u_{xx}(x,t) &&\text{for $x\in\Omega = (-1,1)$, $t\in(0,T]$,}
# \\
# u(-1,t) &= u(1,t) = 0, &&\text{for $t\in[0,T]$,}\\
# u(x,0) &= u_0(x), &&\text{for $x\in\Omega = (-1,1)$.}\\
# \end{align}
#
# The weak formulation of the problem is to find $u(\cdot,t)\in V$ such that
# $$
# \int_{\Omega} u_t(x,t)v(x)dx = - \int_{\Omega} u_x(x,t)v_x(x)dx
# $$
# for all $v\in V$.

# %% [markdown]
# ## 2.1 Spatial finite element approximation

# %% [markdown]
# We consider now the set of $m+2$ points $x_j = -1 + jh$, $h = 2/(m+1)$, for $j=0,\ldots,m+1$, which yields 
#  a discretization grid of $[-1,1]$ and the associated basis functions $x\mapsto\phi_j(x)$, $j=1\ldots,m$.
#
# We seek a finite element approximation of $u$
# $$
# u_h(x,t)
# = \sum_{j=1}^m u_j(t)\phi_j(x)
# $$
#
# that verifies the approximate problem
#
# $$
# \int_{\Omega} \partial_t u_h(x,t)\phi_i(x)dx 
# =
# - \int_{\Omega} \partial_x u_h (x,t)\frac{d \phi_i}{dx}(x)dx
# \quad \text{for all $i=1,\ldots,m$.}
#  $$
#
# By substituting the expression of $u_h$  we get the following system of ordinary differential equations
#
# $$
# \sum_{j=1}^m 
# \frac{d u_j}{dt}(t)
# \int_{\Omega} \phi_i(x)\phi_j(x)dx
# =
# -
#  \sum_{j=1}^m
#  u_j(t)
# \int_{\Omega} \phi_i'(x)\phi_j'(x)dx
# \quad \text{for all $i=1,\ldots,m$.}
# $$ 
#
# As previously, we set
# $$
# A_{ij} = \int_{\Omega} \phi_i'(x)\phi_j'(x)dx,
# \quad
# M_{ij} = \int_{\Omega} \phi_i(x)\phi_j(x)dx
# ,
# $$
# so that the system of ODEs reads
# $$
# M\frac{d U (t)}{dt}  = - A U (t),
# $$
# where 
# $$
# U(t) = (u_1(t),\ldots,u_m(t))^T,
# \quad A = (A_{ij})_{1\leq i,j \leq m}
# \quad M = (M_{ij})_{1\leq i,j\leq m}.
# $$
#
# We recall that if $T$ is the $m \times m$ tridiagonal matrix defined by $T_{i,i+1} = T_{i+1,i} = 1$ for $i=1,\ldots,m-1$ and $T_{i,j} = 0$ otherwise,
#
# $$
# T =
# \begin{pmatrix}
# 0 & 1 & 0 & \cdots & 0\\
# 1 & 0 & 1 & \ddots & \vdots\\
# 0 & \ddots & \ddots & \ddots & 0\\
# \vdots & \ddots & 1 & 0 & 1\\
# 0 & \cdots & 0 & 1 & 0\\
# \end{pmatrix},
# $$
#
# then the matrices $A$ and $M$ can be expressed as 
# $$
# A =\frac{1}{h}
# \left(
# 2 I - T
# \right)
# ,
# \qquad
# M =
# \frac{h}{3}
# \left(
# 2I + \frac{1}{2}T
# \right)
# .
# $$

# %% [markdown]
# ## Exercise
#
# Assemble the matrix A and M.
# %%
def assemble_matrices(m, h):
    return A, M

# %% [markdown]
# ### Validation

# %%
m = 10
h = 2/(m + 1)
k = h**2/6 # comment

A, M = assemble_matrices(m, h)
print(A.todense()) # the first line of K is [11.  -5.5  0.   0.   0.   0.   0.   0.   0.   0. ]
print(M.todense()) # the first line of M is [0.12121212 0.03030303 0. 0. 0. 0. 0. 0. 0. 0. ]


# %% [markdown]
# ## 2.2 Time integration with forward Euler (one-step Adams-Bashforth AB1)

# %% [markdown]
# In this section we propose to approximate the ODE system with forward Euler/one-step Adams-Bashforth.
#
# Let $k>0$ be the time step, we use the time discretization
# $$
# k = T/(N+1)
# ,\quad
# t_n = nk
# ,\quad
# n=0,\ldots,N+1.
# $$
#
# We denote $U^n$ as an approximation of $U(t^n)$.

# %% [markdown]
# ## Exercise (setup)
#
# Show that the update from $U^{n}$ to $U^{n+1}$ takes the form
# $$
# U^{n+1} = U^n - L U^n,
# $$
# where $L$ is a matrix to be determined using $A$ and $M$.

# %% [markdown]
# ## Exercise (implementation)
#
# Complete the following function that computes the values of $(U^0,U^1,\ldots)$ with forward Euler (AB1) for all $n\in\mathbb{N}$ such that $0\leq t^n = n k \leq T$, given a time step $k>0$. The function returns a list of $U^n$ and $t^n$ for every $n$ that is a multiple of 100.
#
# *Hint:* Handle the matrix inversion by solving linear systems using the `spsolve()` scipy routine.

# %%
def heat_AB1(k, h, T, m, u0):
    xx = np.linspace(-1, 1, m + 2) # computation grid
    all_U = []
    all_t = []
    U = u0(xx)
    U = U[1:-1]
    A, M = None, None # To implement: assemble matrices
    step = 0
    t = 0
    all_U.append(U)
    all_t.append(t)
    start = time.time() 
    while t < T:
        U = None # To implement: update the value of U
        t = t + k
        step = step + 1
        if step % 100 == 0:
                all_U.append(U)
                all_t.append(t)
    end = time.time()
    print(f'Time  (tstep): {end-start:.5f}s')
    return all_t, all_U


# %% [markdown]
# ### Validation

# %%
# Setup (problem):
u0 = lambda x: (1-x**2)*np.exp(-10*x**2) # initial condition

# Setup (discretization):
m = 100
h = 2/(m + 1)
k = h**2/6
T = 1000*k
xx = np.linspace(-1, 1, m + 2) # computation grid
xeval = np.linspace(-1, 1, 1000) # evaluation grid

# Solve the heat equation:
all_t, all_U = heat_AB1(k, h, T, m, u0)

# Evaluate the FEM approximation over the evaluation grid:
start = time.time()
def eval_fem(all_U: list, xeval: np.ndarray)-> list:
    n = all_U[0].shape[0]
    J=np.arange(1, n + 1)
    all_Ueval = []
    for U in all_U:
        I, S = np.meshgrid(J, xeval)
        Ueval = Phi(I, S) @ U
        all_Ueval.append(Ueval)    
    return all_Ueval
all_Ueval = eval_fem(all_U, xeval)
end = time.time()
print(f'Time   (eval): {end-start:.5f}s')

# Compute reference solution:
start = time.time() 
all_Uex = [heat_eq_exact(xeval, t, u0) for t in all_t]
end = time.time()
print(f'Time    (ref): {end-start:.5f}s')

# Plot the fem approximate solution with the reference solution:
plot_series(xeval, all_t, all_Ueval, "u_h", all_Uex, "u_ref", "Heat equation (forward Euler)")


# %% [markdown]
# ## 2.3 Time integration with backward Euler (one-step Adams-Moulton AM1)

# %% [markdown]
# In this section we propose to approximate the ODE system with backward Euler/one-step Adams-Moulton.

# %% [markdown]
# ## Exercise (setup)
#
# Show that the update from $U^{n}$ to $U^{n+1}$ takes the form
# $$
# L' U^{n+1} = M U^n,
# $$
# where $L'$ is a matrix to be determined using $A$ and $M$.

# %% [markdown]
# ## Exercise (implementation)
#
# Complete the following function that computes the values of $(U^0,U^1,\ldots)$ with backward Euler (AM1) for all $n\in\mathbb{N}$ such that $0\leq t^n = n k \leq T$, given a time step $k>0$. The function returns a list of $U^n$ and $t^n$ for every $n$ that is a multiple of 100.
#
# *Hint:* Handle the matrix inversion by solving linear systems using the `spsolve()` scipy routine.

# %%
def heat_AM1(k, h, T, m, u0):
    xx = np.linspace(-1, 1, m + 2) # computation grid
    all_U = []
    all_t = []
    U = u0(xx)
    U = U[1:-1]
    A, M = None, None # To implement: assemble matrices
    step = 0
    t = 0
    all_U.append(U)
    all_t.append(t)
    start = time.time() 
    while t < T:
        U = None # To implement: update the value of U
        t = t + k
        step = step + 1
        if step % 100 == 0:
                all_U.append(U)
                all_t.append(t)
    end = time.time()
    print(f'Time  (tstep): {end-start:.5f}s')
    return all_t, all_U


# %% [markdown]
# ### Validation

# %%
# Setup (problem):
u0 = lambda x: (1-x**2)*np.exp(-10*x**2) # initial condition

# Setup (discretization):
m = 100
h = 2/(m + 1)
k = h**2/6
T = 1000*k
xx = np.linspace(-1, 1, m + 2) # computation grid
xeval = np.linspace(-1, 1, 1000) # evaluation grid

# Solve the heat equation:
all_t, all_U = heat_AM1(k, h, T, m, u0)

# Evaluate the FEM approximation over the evaluation grid:
start = time.time()
all_Ueval = eval_fem(all_U, xeval)
end = time.time()
print(f'Time   (eval): {end-start:.5f}s')

# Compute reference solution:
start = time.time() 
all_Uex = [heat_eq_exact(xeval, t, u0) for t in all_t]
end = time.time()
print(f'Time    (ref): {end-start:.5f}s')

# Plot the fem approximate solution with the reference solution:
plot_series(xeval, all_t, all_Ueval, "u_h", all_Uex, "u_ref", "Heat equation (backward Euler)")

# %% [markdown]
# ## 2.4 Spectrum analysis

# %% [markdown]
# Let us examine the spectrum of the matrices T, A and M previously computed and compare them with the analytical formulas:
#
# $$
# \begin{align*}
# \operatorname{sp}(T) 
# &=
# \left\{
# -2 + 4 \cos^2
# \left(
# \frac{j\pi}{2(m+1)}
# \right)
# ,\
# j=1,\ldots,m
# \right\},
# \\
# \operatorname{sp}(A) 
# &=
# \left\{
# \frac{4}{h}
# \sin^2
# \left(
# \frac{j\pi}{2(m+1)}
# \right)
# ,\
# j=1,\ldots,m
# \right\},
# \\
# \operatorname{sp}(M) 
# &=
# \left\{
# \frac{h}{3}
# \left(
# 1+
# 2\cos^2
# \left(
# \frac{j\pi}{2(m+1)}
# \right)
# \right)
# ,\
# j=1,\ldots,m
# \right\}.
# \end{align*}
# $$

# %% [markdown]
# ## Exercise
#
# Compute the spectrum of T, A, M as a sorted numpy array and compare the result with the analytical formulas.
#
# *Hints:*
# * transform the sparse matrice into dense matrices and use the numpy `eigvals()` function;
# * sort the element of a numpy arrays using the numpy `sort()` function.

# %%
m = 100
h = 2/(m + 1)
T, A, M = None, None, None # To implement: assemble the matrices T, A, M

# eigenvalues of T
egval_T = None # To implement: compute the eigenvalues of the numpy matrix T
egval_T_ref = None # To implement: compute analytical expression of the eigenvalues of T
#print(np.linalg.norm(egval_T - egval_T_ref))

# eigenvalues of A
egval_A = None # To implement: compute the eigenvalues of the numpy matrix K
egval_A_ref = None # To implement: compute analytical expression of the eigenvalues of K
#print(np.linalg.norm(egval_A - egval_A_ref))

# eigenvalues of M
egval_M = None # To implement: compute the eigenvalues of the numpy matrix M
egval_M_ref = None # To implement: compute analytical expression of the eigenvalues of M
#print(np.linalg.norm(egval_M - egval_M_ref))

# %% [markdown]
# ## 2.5 Stability analysis

# %% [markdown]
# We recall that the spectral radius $\sigma(R)$ of a square matrix $R$  is $\sigma(R) = \max_{\lambda\in\operatorname{sp}(R)}|\lambda|$. It is easy to show that
#
# $$
# \sigma(A) \leq 4/h, \quad \sigma(M^{-1}) \leq 3/h,
# $$
#
# and that
#
# $$
# \sigma(M^{-1}A) \leq \sigma(M^{-1})\sigma(A) \leq 12/h^2.
# $$
#
# The resulting stability restriction for the time-step $k$ for forward Euler reads $\vert k\lambda(-M^{-1}A) + 1\vert\leq1$, that is,
#
# $$
# -1 \leq 1 - 12k/h^2 \leq 1,
# $$
#
# which yields $k\leq h^2/6$. 
#
# For backward Euler, $\vert k\lambda(-M^{-1}A) - 1\vert \geq1$ does not give any restriction.


# %% [markdown]
# ## Exercise
# Recompute the solution of sections 2.2 and 2.3 with $m=100$ and time-step $k=2h^2/6$ to illustrate numerical instability.

# %% [markdown]
# # 3 Allen-Cahn equation with homogeneous Dirichlet boundary conditions

# %% [markdown]
# Let $\varepsilon>0$. We consider the one-dimensional Allen-Cahn equation with homogeneous Dirichlet boundary conditions;
#
# \begin{align}
# u_t(x,t) &= \varepsilon u_{xx}(x,t) + u(x,t) - u(x,t)^3 &&\text{for $x\in\Omega = (-1,1)$, $t\in(0,T]$,}
# \\
# u(-1,t) &= u(1,t) = 0, &&\text{for $t\in[0,T]$,}\\
# u(x,0) &= u_0(x), &&\text{for $x\in\Omega = (-1,1)$.}\\
# \end{align}
#
# The weak formulation of the problem is to find $u(\cdot,t)\in V$ such that
#
# $$
# \int_{\Omega} u_t(x,t)v(x)dx 
# =
# - \int_{\Omega} u_x(x,t)v_x(x)dx
# + \int_{\Omega} u(x,t)v(x)dx
# + \int_{\Omega} u^3(x,t)v(x)dx
# $$
# for all $v\in V$.

# %% [markdown]
# ## 3.1 Spatial finite element approximation

# %% [markdown]
# We consider now the set of $n+2$ points $x_j = -1 + jh$, $h = 2/(m+1)$, for $j=0,\ldots,m+1$, which yields a discretization grid of $[-1,1]$ and associated basis functions $x\mapsto\phi_j(x)$, $j=1\ldots,m$.

# %% [markdown]
# We seek a finite element approximation of $u$ 
#
# $$
# u_h(x,t)
# = \sum_{j=1}^m u_j(t)\phi_j(x)
# $$
#
# that verifies the approximate problem
#
# $$
# \int_{\Omega} \partial_t u_h(x,t)\phi_i(x)dx 
# =
# -
# \varepsilon
# \int_{\Omega} \partial_x u_h (x,t)\frac{d \phi_i}{dx}(x)dx
# +
# \int_{\Omega} u_h (x,t)\phi_i(x)dx
# -
# R_i
# \quad \text{for all $i=1,\ldots,m$,}
# $$
#
# where
#
# $$
# \text{$R_i$ is an approximation of}
# \quad
# \int_{\Omega} u_h (x,t)^3 \phi_i(x)dx
# \quad \text{for all $i=1,\ldots,m$.}
# $$

# %% [markdown]
# We note again
# $$
# A_{ij} = \int_{\Omega} \phi_i'(x)\phi_j'(x)dx,
# \quad
# M_{ij} = \int_{\Omega} \phi_i(x)\phi_j(x)dx
# ,
# \quad
# U(t) = (u_1(t),\ldots,u_m(t))^T,
# \quad
# A= (A_{ij})_{1\leq i,j \leq m}
# \quad
# M = (M_{ij})_{1\leq i,j\leq m}.
# $$


# %% [markdown]
# ## Exercise
#
# We choose to set
#
# $$
# R_i=
# \sum_{j=1}^m
# \int_{\Omega} u_j (t)^3 \phi_j(x)\phi_i(x)dx
# \quad \text{for all $i=1,\ldots,m$.}
# $$
#
# Using the Hadamard product $U(t)\odot U(t) \odot U(t) = (u_1(t)^3,\cdots,u_m(t)^3)^T$, $A$ and $M$, write the system of ordinary differential equations verified by $t\mapsto U(t)$.

# %% [markdown]
# ## 3.2 Time integration with forward Euler

# %% [markdown]
# Let $k>0$ be the time step, we use the time discretization
# $$
# k = T/(N+1)
# ,\quad
# t_n = nk
# ,\quad
# n=0,\ldots,N+1.
# $$
#
# We denote $U^n$ as an approximation of $U(t^n)$.


# %% [markdown]
# ## Exercise (setup)
#
# Perform a time discretization of the ODE system using the forward Euler method. Show that it takes the form
# $$
# U^{n+1} = U^{n} + k V^{n}
# ,
# $$
# where $V^{n}$ is to be determined using $M$, $A$, $U^{n}$ and $U^{n}\odot U^{n} \odot U^{n}$.

# %% [markdown]
# ## Exercise (implementation)
#
# Complete the following function that computes the values of $(U^0,U^1,\ldots)$ with forward Euler (AB1) for all $n\in\mathbb{N}$ such that $0\leq t^n = n k \leq T$, given a time step $k>0$. The function returns a list of $U^n$ and $t^n$ for every $n$ that is a multiple of 100.
#
# *Hint:* Handle the matrix inversion by solving linear systems using the `spsolve()` scipy routine.

# %%
def Allen_Cahn_AB1(k, h, T, m, u0, eps):
    xx = np.linspace(-1, 1, m + 2) # computation grid
    all_U = []
    all_t = []
    U = u0(xx)
    U = U[1:-1]
    A, M = None, None # To implement: assemble matrices
    step = 0
    t = 0
    all_U.append(U)
    all_t.append(t)
    start = time.time() 
    while t < T:
        U = None # To implement: update the value of U
        t = t + k
        step = step + 1
        if step % 100 == 0:
                all_U.append(U)
                all_t.append(t)
    end = time.time()
    print(f'Time  (tstep): {end-start:.5f}s')
    return all_t, all_U


# %% [markdown]
# ### Validation
#
#  Solve the Allen-Cahn equation with the Adam-Bashforth order 1 scheme.

# %%
# Setup (problem):
eps = 0.01
u0 = lambda x: np.exp(-100*(x+0.5)**2) - np.exp(-100*(x-0.5)**2) # initial condition

# Setup (discretization):
m = 100
h = 2/(m + 1)
xx = np.linspace(-1, 1, m + 2) # computation grid
xeval = np.linspace(-1, 1, 1000) # evaluation grid
k = 1/eps*h**2/6
T = 1000*k
N = int(T/k - 1)

# Solve the Allen Cahn equation:
all_t, all_U = Allen_Cahn_AB1(k, h, T, m, u0, eps)

# Evaluate the FEM approximation over the evaluation grid:
all_Ueval = eval_fem(all_U, xeval)

# Import reference solution at final time:
uex = np.loadtxt('uex-ac.txt')

# Plot the fem approximate solution:
plot_series(xeval, all_t, all_Ueval, "u_h", title="Allen-Cahn (forward Euler)")

# Compare the approximate solution with the reference solution final time:
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=xeval, y=all_Ueval[-1], mode='markers', name='uh'))
fig.add_trace(go.Scatter(x=xeval, y=uex, mode='lines', name='uex'))
fig.update_layout(title='Allen-Cahn (forward Euler)', xaxis_title='x', yaxis_title='u')
fig.show()

# Evaluation of the L-inf relative error at the final time with respect to the reference solution:
err = np.linalg.norm(all_Ueval[-1] - uex,np.inf) / np.linalg.norm(uex,np.inf)
print(f'Error at final time: {err:.2e}')
