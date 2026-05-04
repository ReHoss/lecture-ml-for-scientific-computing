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

# %% [markdown]
# <div style="text-align: center;">
# <strong>Machine Learning for Scientific Computing and Numerical Analysis - Practical 1</strong>
# </div>
# <div style="text-align: center;">
# <i>Linear regression, nonlinear regression, and trigonometric interpolation</i>
# </div>
# <div style="text-align: center;">
# </br>
# <p>Victorita Dolean, Loïc Gouarin, Rémy Hosseinkhan, Hadrien Montanelli
# </br>
# 2025-2026
# </p>

# %% [markdown]
# In this first PC, we will explore various regression methods using datasets that we generate ourselves. The goal is to understand how the characteristics of these datasets influence the solution to the supervised learning problem.

# %% [markdown]
# # Import the Python modules used

# %%
import numpy as np
from scipy.stats import multivariate_normal

# Plotting libraries
import matplotlib.pyplot as plt
import plotly
import ipywidgets
import anywidget


# %% [markdown]
# # Datasets
# We will start by presenting several datasets that will be used throughout this notebook. You can add your own dataset by adding an entry in the following function.


# %%
def get_dataset(datatype, n, d, eta, random_gen=np.random.rand):
    np.random.seed(42)
    x = -1 + 2 * random_gen(n, d)
    if datatype == "linear":
        w_linear = np.ones(d)
        y = x @ w_linear + eta * random_gen(n)
    elif datatype == "logist":
        w_logistic = 3 * np.ones(d)
        y = 1 / (1 + np.exp(-x @ w_logistic)) + eta * random_gen(n)
    elif datatype == "sphere":
        y = np.linalg.norm(x, axis=1) + eta * random_gen(n)
    elif datatype == "normal":
        mean = np.zeros(d)
        covariance = np.eye(d) / d
        mvn = multivariate_normal(mean=mean, cov=covariance)
        y = mvn.logpdf(x) + eta * random_gen(n)
    y /= np.max(np.abs(y))
    return x, y


# %% [markdown]
# Let’s examine the data and observe how a random perturbation impacts its quality. To make it possible to visualize the solutions in 2D, we will assume for now that there is only one weight $w$ (i.e., dimension $d=1$). This will no longer be the case when we are interested in solving the problem.

# %%
from plot_dataset import create_plot_dataset_container

create_plot_dataset_container(get_dataset)


# %% [markdown]
# # 1. Linear least squares
#
# As explained in the first lecture, we want to minimize
#
# $$
# \Vert X\bm{w} - \bm{y}\Vert_2^2,
# $$
#
# where the rows of $X$ represent one sample. Suppose we have $n$ samples and we try to find $d$ weights, then the $X$ matrix is of size $(n , d)$. The optimal weights are obtained by solving the normal equations
#
# $$
# X^TX\bm{w} = X^T\bm{y}.
# $$
#
# Various methods are available for solving this linear system (QR, SVD, iterative methods, ...). In the following, we will focus on the QR factorization $X=QR$, where $Q$ is an orthogonal matrix and $R$ is an upper-triangular matrix.
#
# The construction of $Q$ can be done using the Gram-Schmidt algorithm, which can be described as follows.
#
# #### Gram-Schmidt Algorithm
#
# **Input** A list of vectors $\{v_1,v_2,\cdots,v_d\}$ corresponding to the columns of $X$.
#
# **Output** A list of orthonormal vectors $\{q_1,q_2,\cdots,q_d\}$ correponsding to the columns of $Q$.
#
# **Steps**
#
# 1. Initialize an empty matrix $Q$ to store the orthonormal vectors.
# 2. For each vector $v_i$, $i=1,2,\ldots,d$:
#    - Start with $q_i = v_i$.
#    - Subtract the projections of $q_i$ onto all previously computed orthogonal vectors $q_j$ (for $j=1,2,\cdots,i−1)$.
#    - Normalize $q_i$ by dividing by its norm.
# 3. Append each orthonormal vector $q_i$ to $Q$.
# 4. Return the matrix $Q$ of orthonormal vectors.
#
# **Notes**
#
# - The projection of $v_i$ onto $q_j$ is calculated as $P_{q_j}(v_i)=(v_i^Tq_j)q_j$.
#
# - After orthogonalizing $v_i$, it can be normalized by dividing by its norm to get an orthonormal vector.
#

# %% [markdown]
# In the next cell, we provide a partial implementation of the Gram-Schmidt algorithm, leaving out the projection step. *(Note: This simple implementation is numerically unstable and may perform poorly when the matrix is ill-conditioned. In such cases, a modified Gram-Schmidt algorithm should be used.)*


# %%
def gram_schmidt(A):
    Q = np.empty_like(A)
    for i in range(A.shape[1]):
        q = np.copy(A[:, i])
        for j in range(i):
            # Part to code

            
            # End part to code
        q /= np.sqrt(q.T @ q)
        Q[:, i] = q
    return Q

# %% [markdown]
# ## Exercise
#
# Write the second step of the algorithm, specifically the part where you have *Part to code*. In other words: *Subtract the projections of $v_i$ onto all previously computed orthonormal vectors $q_j$ (for $j=1,2,\cdots,i−1)$.*
#
# Validate your algorithm using the following matrix
#
# $$
# A = \left[
# \begin{array}{ccc}
# 1 & 1 & 1 \\
# -1 & 0 & 1 \\
# 1 & 1 & 2
# \end{array}
# \right].
# $$
#
# You can use the `qr` method provided in `numpy.linalg` to check the result.

# %% [markdown]
# ## Exercise
#
# Write a `QR_factorization` function, which outputs both $Q$ and $R$.

# %% [markdown]
# The QR factorization can be computed in $\mathcal{O}(2nd^2)$ operations with Gram-Schmidt orthogonalization.
#
# ## Exercise
#
# Verify the complexity of the algorithm using the following function.

# %%
import time


def computation_time(n, d):
    exec_time = []
    for i in n:
        for j in d:
            A = np.random.rand(i, j)
            t0 = time.time()
            QR_factorization(A)
            t1 = time.time()
            exec_time.append(t1 - t0)
    return exec_time

# %% [markdown]
# We can now solve the normal equations via
#
# $$
# R\bm{w} = Q^T\bm{y}.
# $$
#
# Since $R$ is an upper-triangular matrix, we can compute $\bm{w}$ using back substitution. The function below is a step-by-step solution for solving linear least sqaures with QR factorization (refer to *Algorithm 1* in the first lecture).


# %%
def solve_linear_least_squares(X, y):
    Q, R = QR_factorization(X)
    b = Q.T @ y
    w = np.empty(X.shape[1])
    for i in range(X.shape[1] - 1, -1, -1):
        w[i] = (b[i] - np.sum(w[i + 1 :] * R[i, i + 1 :])) / R[i, i]
    return w


def solve_linear_least_squares_np(X, y):  # implementation with built-in numpy functions
    Q, R = np.linalg.qr(X)
    w = np.linalg.solve(R, Q.T @ y)
    return w


X = np.random.rand(1000, 100)
y = np.ones(1000)
w = solve_linear_least_squares(X, y)
w_np = solve_linear_least_squares_np(X, y)
print(np.linalg.norm(w - w_np))

# %% [markdown]
# The supervised learning problem can now be solved using the previous function. Next, we will examine the robustness of this algorithm in determining the optimal weights for the datasets under consideration.

# %%
from plot_dataset import create_plot_error_container

create_plot_error_container(get_dataset, solve_linear_least_squares)

# %% [markdown]
# ## Exercise
#
# What do you observe? Comment.

# %% [markdown]
# # 2. Nonlinear least squares
#
# In the previous section, we explored how to solve a linear regression problem. We found that the solution works well for linear data but fails for nonlinear data. To achieve good results with nonlinear data, an additional step is necessary.
#
# Various methods exist to solve nonlinear regression problems. We compare two of them:
#
# - Gauss-Newton's method with QR factorization;
# - Newton's method.
#
# Both methods aim to minimize
#
# $$
# \Vert\hat{\bm{y}}(\bm{w}) - \bm{y}\Vert_2^2 \qquad \text{where} \qquad \hat{\bm{y}}(\bm{w}) = (\hat{f}(\bm{x}_1;\bm{w}),\ldots,\hat{f}(\bm{x}_n;\bm{w}))^T,
# $$
#
# for some function model $\hat{f}$. We start with Gauss-Newton's method with QR factorization.

# %% [markdown]
# ## 2.1 Gauss-Newton's method with QR
#
# #### Gauss-Newton's Method with QR
#
# **Inputs**
#
# - $\hat{f}$: the function model.
# - $J$: the Jacobian of $\hat{f}$ with respect to $\bm{w}$.
# - $\{\bm{x}_i,y_i\}_{i=1}^n$: the data points.
# - $\bm{w^0}$: the initial guess.
#
# **Output** The optimal weights $\bm{w}$.
#
# **Steps**
#
# 1. Compute the Jacobian $J^k=J(\bm{w}^k)$ and evaluate the prediction $\hat{\bm{y}}(\bm{w}^k)$.
# 2. Compute the vector $\bm{r}^k = \hat{\bm{y}}(\bm{w}^k) - \bm{y}$ .
# 2. Solve the linearized normal equations $(J^k)^TJ^k\bm{s}^k=−(J^k)^T\bm{r}^k$ using QR decomposition.
# 3. Update $\bm{w}^{k+1} = \bm{w}^k + \bm{s}^k$.
# 4. Repeat until convergence.
#
# ## Exercise
#
# Complete the following routine to implement Gauss-Newton's method with QR decomposition.


# %%
def gauss_newton(fhat, Jacobian, X, y, w0, *, tol=1e-6, maxit=100, verbose=False):
    w = w0
    history = []
    for it in range(maxit):
        J = Jacobian(X, w)
        yhat = fhat(X, w)
        # Part to code

        
        # End part to code
        w += s
        error = np.square(fhat(X, w) - y).mean()
        history.append(error)
        if verbose:
            print(f"iteration {it}: error = {error}")
        if error < tol or np.linalg.norm(s) < tol:
            break
    return w, it, history

# %% [markdown]
# As you have noticed, you have to give the function and the Jacobian to the solve nonlinear least squares problem. We choose the logistic data set and the logistic function as a model.

# %%
dataset = "logist"
eta = 1e-2
n = 1000
d = 100
X, y = get_dataset(dataset, n, d, eta)
fhat = lambda X, w: 1 / (1 + np.exp(-X @ w))

# %% [markdown]
# ## Exercise
#
# Compute the Jacobian and perform *Algorithm 2*.

# %% [markdown]
# ## 2.2 Newton's method
#
# We continue with Newton's method. We recall that Newton's method for minimizing $\mathcal{E}_T(\bm{w})=\Vert\hat{\bm{y}}(\bm{w}) - \bm{y}\Vert_2^2$ tries to solve $\bm{g}(\bm{w})=0$, where
#
# $$
# \bm{g}(\bm{w}) = J^T(\bm{w})r(\bm{w}), \qquad \bm{r}(\bm{w}) = \hat{\bm{y}}(\bm{w}) - \bm{y},
# $$
#
# is the gradient of $\mathcal{E}_T(\bm{w})$ (up to a scaling factor). To do so, it uses the Hessian matrix $H$ of $\mathcal{E}_T(\bm{w})$.
#
# #### Newton's Method
#
# **Inputs**
#
# - $\hat{f}$: the function model.
# - $\bm{g}$: the gradient of $\mathcal{E}_T(\bm{w})$ with respect to $\bm{w}$.
# - $H$: the Hessian of $\mathcal{E}_T(\bm{w})$ with respect to $\bm{w}$.
# - $\{\bm{x}_i,y_i\}_{i=1}^n$: the data points.
# - $\bm{w^0}$: the initial guess.
#
# **Output** The optimal weights $\bm{w}$.
#
# **Steps**
#
# 1. Compute the gradient $\bm{g}^k=\bm{g}(\bm{w}^k)$ and the Hessian $H^k=H(\bm{w}^k)$.
# 2. Solve $H^k\bm{s}^k=−\bm{g}^k$.
# 3. Update $\bm{w}^{k+1} = \bm{w}^k + \bm{s}^k$.
# 4. Repeat until convergence.
#
# ## Exercise
#
# Complete the following routine to implement Newton's method.


# %%
def newton(fhat, gradient, Hessian, X, y, w0, *, tol=1e-6, maxit=200, verbose=False):
    w = w0
    for it in range(maxit):
        g = gradient(X, y, w)
        H = Hessian(X, w)
        # Part to code
        
        # End part to code
        w += s
        error = np.square(fhat(X, w) - y).mean()
        if verbose:
            print(f"iteration {it}: error = {error}")
        if error < tol or np.linalg.norm(s) < tol:
            break
    return w, it

# %% [markdown]
# Here, you have to provide the gradient and the Hessian matrix.
#
# ## Exercise
#
# Compute the gradient and the Hessian matrix and perform *Algorithm 3*. Comment.

# %% [markdown]
# The number of iterations may be high because the Hessian matrix is not always positive definite away from a minimum. One way to address this, which results in a method equivalent to Gauss-Newton’s method, is to replace to replace $H$ by $J^TJ$.
#
# ## Exercise
#
# Use $J^TJ$ instead of $H$ and perform *Algorithm 3* again. What do you observe?

# %% [markdown]
# # 3. Trigonometric interpolation
#
# As discussed in the lecture, linear and nonlinear regression can approximate general functions, but for **periodic** data, trigonometric polynomials are the natural choice.
#
# We seek a trigonometric polynomial $f_\theta:[0,2\pi]\to\mathbb{C}$ of the form
#
# $$
# f_\theta(x) = \sum_{k=-(n-1)/2}^{(n-1)/2} \theta_k e^{ikx},
# $$
#
# where the coefficients $\theta_k$ are computed using the Fast Fourier Transform (FFT).
#
# ## Exercise
#
# Implement the `trigonometric_interpolation` function.
# 1. Compute the coefficients $\theta_k$ using `np.fft.fft`. Don't forget to normalize by $1/n$.
# 2. Reconstruct the signal on `x_plot` using the formula above. You can use `np.fft.fftfreq` to get the correct frequencies $k$.


# %%
def trigonometric_interpolation(x_nodes, y_nodes, x_plot):
    n = len(y_nodes)
    # Part to code


    
    # End part to code
    return y_plot

# %% [markdown]
# Let's test this on the function $f(x) = \sin(x) + 0.3\sin(3x)$ used in the lecture.

# %%
n = 9  # Number of sample points (odd)
x_nodes = np.linspace(0, 2 * np.pi, n, endpoint=False)
y_nodes = np.sin(x_nodes) + 0.3 * np.sin(3 * x_nodes)

x_plot = np.linspace(0, 2 * np.pi, 200)
y_true = np.sin(x_plot) + 0.3 * np.sin(3 * x_plot)
y_interp = trigonometric_interpolation(x_nodes, y_nodes, x_plot)

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_true, "k--", label="True function")
plt.plot(x_nodes, y_nodes, "bo", label="Samples")
plt.plot(x_plot, y_interp.real, "r-", label="Trigonometric Interpolant")
plt.legend()
plt.title("Trigonometric Interpolation")
plt.grid(True)
plt.show()

# %% [markdown]
# # 4. Understanding Errors and Data Quality
#
# As discussed in Chapter 1 (Section 3), the error in supervised learning can be decomposed into three components:
#
# $$
# \mathcal{E}_G(\boldsymbol{\theta}) \le \underbrace{\mathcal{E}_G(\boldsymbol{\theta}^*_G)}_{\text{approximation error}} + \underbrace{2\sup_{\boldsymbol{\theta}'\in\Theta} |\mathcal{E}_G(\boldsymbol{\theta}') - \mathcal{E}_T(\boldsymbol{\theta}')|}_{\text{generalization gap}} + \underbrace{\mathcal{E}_T(\boldsymbol{\theta}) - \mathcal{E}_T(\boldsymbol{\theta}^*_T)}_{\text{optimization error}}
# $$
#
# This decomposition highlights three fundamental sources of error:
#
# 1.  **Approximation error**: This term, $\mathcal{E}_G(\boldsymbol{\theta}^*_G)$, represents the best possible performance of our model class $\Theta$. It measures the "distance" between the true function $f$ and the best function in our hypothesis space. If our model is too simple (e.g., a linear model for complex data), this error will be large. This is often referred to as **bias**.
#
# 2.  **Generalization gap**: This term measures the discrepancy between the training error $\mathcal{E}_T$ and the generalization error $\mathcal{E}_G$. It tells us how well the performance on the training set predicts the performance on unseen data. A large gap usually indicates **overfitting**, where the model memorizes the training data but fails to generalize. This is closely related to **variance** and data quality.
#
# 3.  **Optimization error**: This term, $\mathcal{E}_T(\boldsymbol{\theta}) - \mathcal{E}_T(\boldsymbol{\theta}^*_T)$, arises when we fail to find the global minimum of the training loss. This can happen if the optimization algorithm stops too early, gets stuck in a local minimum, or converges slowly.
#
# Let's illustrate these errors with concrete examples.

# %% [markdown]
# ## 4.1 Approximation Error
#
# This error comes from the limited expressivity of the model class. If we try to fit a highly nonlinear function with a linear model, the best possible model will still have a high error.
#
# Consider $f(x) = \sin(2\pi x)$ on $[0, 1]$.

# %%
# Generate data
n = 100
x = np.linspace(0, 1, n)
y = np.sin(2 * np.pi * x)

# Linear model (y = wx + b)
X = np.vstack([x, np.ones(n)]).T
w_linear = solve_linear_least_squares(X, y)
y_linear = X @ w_linear

# Polynomial model (degree 3)
X_poly = np.vstack([x**3, x**2, x, np.ones(n)]).T
w_poly = solve_linear_least_squares(X_poly, y)
y_poly = X_poly @ w_poly

plt.figure(figsize=(10, 6))
plt.plot(x, y, "k--", label="True function", linewidth=2)
plt.plot(x, y_linear, label="Linear Model (High Approx. Error)")
plt.plot(x, y_poly, label="Polynomial Model (Low Approx. Error)")
plt.legend()
plt.title("Approximation Error: Linear vs Polynomial Model")
plt.grid(True)
plt.show()

print(f"Linear Model MSE: {np.mean((y - y_linear)**2):.4f}")
print(f"Polynomial Model MSE: {np.mean((y - y_poly)**2):.4f}")

# %% [markdown]
# ## 4.2 Optimization Error
#
# This error arises when the optimization algorithm fails to find the global minimum of the training loss. This can happen if we stop too early or get stuck in a local minimum.
#
# Let's revisit the logistic regression example but vary the number of iterations.

# %%
dataset = "logist"
n = 1000
d = 100
X, y = get_dataset(dataset, n, d, 1e-2)
fhat = lambda X, w: 1 / (1 + np.exp(-X @ w))
J = lambda X, w: (fhat(X, w) * (1 - fhat(X, w)))[:, np.newaxis] * X
w0 = 1e-2 * np.random.rand(d)

# Run Gauss-Newton until convergence
w_opt, it_opt, history = gauss_newton(fhat, J, X, y, w0.copy(), maxit=50)

# Plot the training error over iterations
plt.figure(figsize=(10, 6))
plt.plot(range(len(history)), history, "b-o", label="Training Error")
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")
plt.title("Optimization Error: Convergence of Gauss-Newton")
plt.yscale("log")
plt.grid(True)

# Highlight specific points
plt.scatter(0, history[0], color="red", s=100, zorder=5, label="Start (High Error)")
plt.scatter(len(history)-1, history[-1], color="green", s=100, zorder=5, label="Converged (Low Error)")
plt.legend()
plt.show()

print(f"Initial Error: {history[0]:.4f}")
print(f"Error at iteration 5: {history[5] if len(history)>5 else history[-1]:.4f}")
print(f"Final Error (Converged): {history[-1]:.4f}")
print(f"Optimization Error (Early Stopping at it=5): {history[5] - history[-1]:.4f}")

# %% [markdown]
# ## 4.3 Generalization Gap and Data Distribution
#
# The generalization gap measures the difference between training performance (empirical risk under the training/empirical distribution) and performance against the true data distribution.
#
# Consider the function $f(x) = x + 0.1 x^3$.
# *   Near $x=0$, it looks linear ($x^3$ is small).
# *   Far from $x=0$, the nonlinearity dominates.
#
# If we train a linear model only on data from $[-0.5, 0.5]$, it will fit well (low training error). However, it will fail catastrophically when tested on $[-2, 2]$ (high generalization error). Indeed the true distribution with support $[-2,2]$ will be incorrectly approximated by the model trained on the empirical distribution with support $[-0.5,0.5]$.


# %%
def target_func(x):
    return x + 0.5 * x**3


# Training data: concentrated in the "linear" region [-0.5, 0.5]
x_train = np.linspace(-0.5, 0.5, 20)
y_train = target_func(x_train)

# Test data: covers the full domain [-2, 2]
x_test = np.linspace(-2, 2, 100)
y_test = target_func(x_test)

# Train a linear model
X_train_mat = np.vstack([x_train, np.ones(len(x_train))]).T
w_train = solve_linear_least_squares(X_train_mat, y_train)

# Predict
y_train_pred = X_train_mat @ w_train

X_test_mat = np.vstack([x_test, np.ones(len(x_test))]).T
y_test_pred = X_test_mat @ w_train

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x_test, y_test, "k--", label="True function $x + 0.5x^3$", linewidth=2)
plt.plot(x_test, y_test_pred, "r-", label="Linear Model Prediction (Test Domain)")
plt.scatter(x_train, y_train, color="blue", s=50, label="Training Data (Local)")
plt.legend()
plt.title("Generalization Gap: Local Linearity vs Global Nonlinearity")
plt.grid(True)
plt.show()

train_mse = np.mean((y_train - y_train_pred) ** 2)
test_mse = np.mean((y_test - y_test_pred) ** 2)

print(f"Training MSE (on [-0.5, 0.5]): {train_mse:.6f}")
print(f"Test MSE (on [-2, 2]): {test_mse:.6f}")


# %% [markdown]
# This example highlights that **linear regression is not enough** if the global behavior is nonlinear, even if it works locally. It also emphasizes that **data must cover the domain of interest** to ensure good generalization.

# %% [markdown]
# # Conclusion
#
# In this PC, we have explored the three main pillars of classical supervised learning without neural networks:
#
# 1.  **Linear Least Squares** (QR): Optimal for linear relationships, with a closed-form solution.
# 2.  **Nonlinear Regression** (Gauss-Newton/Newton): Necessary for nonlinear models, requiring iterative optimization.
# 3.  **Trigonometric Interpolation** (FFT): The natural choice for periodic data, leveraging the efficiency of the FFT.
#
# We also analyzed the **error decomposition** (approximation, optimization, generalization) and demonstrated how **data quality** (distribution) critically affects the generalization gap.
#
# These methods form the "classical backbone" of Scientific Computing, as introduced in Chapter 1. In the next PC, we will see how Neural Networks can extend these capabilities to even more complex, high-dimensional problems.
