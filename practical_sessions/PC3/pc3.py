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
# <strong>Machine Learning for Scientific Computing and Numerical Analysis - Practical 3</strong>
# </div>
# <div style="text-align: center;">
# <i>Optimization</i>
# </div>
# <div style="text-align: center;">
# </br>
# <p>Victorita Dolean, Loïc Gouarin, Rémy Hosseinkhan, Hadrien Montanelli
# </br>
# 2025-2026
# </p>

# %% [markdown]
# # Introduction

# %% [markdown]
# In this session, we will implement a multilayer perceptron in order to gain a better understanding of the basic algorithms that make up neural networks.
# The forward pass has already been implemented in PC2. We will now focus on the backpropagation algorithm, which allows us to fully implement the training of a neural network.
#
# We recall here the notations used during the lectures and throughout this session.
#
# Let $d_0,\ldots,d_{L+1}$ be positive integers such that $d_0=d$ and $d_{L+1}=1$. An artificial neural network $\hat{f}(\cdot;\bm{w}):\R^d\to\R$ is a composition of artificial neurons, i.e., $\hat{f}(\cdot;\bm{w}) = a^{(L+1)} \circ \ldots \circ \bm{a}^{(0)}$ with
#
# $$
# \begin{align}
# & \bm{a}^{(0)}:\R^d\to\R^d, && \bm{a}^{(0)}(\bm{x}) = \bm{x}, \\
# & \bm{a}^{(\ell)}:\R^{d_{\ell-1}}\to\R^{d_\ell}, && \bm{a}^{(\ell)}(\bm{x}) = \sigma\left(W^{(\ell-1)}\bm{x} + \bm{b}^{(\ell-1)}\right), \quad 1\leq\ell\leq L, \\
# & a^{(L+1)}:\R^{d_L}\to\R, && a^{(L+1)}(\bm{x}) = W^{(L)}\bm{x} + b^{(L)}.
# \end{align}
# $$

# %% [markdown]
# Here, $\sigma$ is the activation function. Various activation functions are used in artificial neural networks. We introduce three of them: sigmoid, hyperbolic tangent, and ReLU.
# These functions were implemented in PC2 (Exercise 1). We provide their implementations below, as the forward pass uses the activation function and backpropagation uses its derivative.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import time


# %%
class sigmoid:
    def __call__(self, x):
        return 1.0 / (1 + np.exp(-x))

    def diff(self, x):
        f = self(x)
        return f * (1 - f)


# %%
class tanh:
    def __call__(self, x):
        return np.tanh(x)

    def diff(self, x):
        f = self(x)
        return 1 - f**2


# %%
class ReLU:
    def __call__(self, x):
        return np.maximum(0, x)

    def diff(self, x):
        grad = np.zeros_like(x)
        grad[x > 0] = 1
        return grad


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
# # 1 Shallow networks

# %% [markdown]
# Let's take this simple example to illustrate the forward pass and backpropagation. We have an input in two dimensions ($d_0=d=2$), followed by a single hidden layer ($L=1$) with three neurons ($d_1=3)$ and finally, a scalar output ($d_2=1$). The forward pass can be written as:
#
# $$
# \begin{aligned}
# &\boldsymbol{a}^{(0)} = (x_1, x_2)^T \in \R^2, \\ \\
# &\boldsymbol{z}^{(0)} = \left[
# \begin{array}{cc}
# W^{(0)}_{1, 1} & W^{(0)}_{1, 2} \\
# W^{(0)}_{2, 1} & W^{(0)}_{2, 2} \\
# W^{(0)}_{3, 1} & W^{(0)}_{3, 2}
# \end{array}
# \right]
# \left(
# \begin{array}{c}
# a^{(0)}_1 \\
# a^{(0)}_2
# \end{array}
# \right)
# +
# \left(
# \begin{array}{c}
# b^{(0)}_1 \\
# b^{(0)}_2 \\
# b^{(0)}_3
# \end{array}
# \right) \in \R^3, \\ \\
# & \boldsymbol{a}^{(1)} = \sigma\left(\boldsymbol{z}^{(0)}\right) \in \R^3, \\ \\
# & a^{(2)} = z^{(1)} = \left[
# \begin{array}{cccc}
# W^{(1)}_{1, 1} & W^{(1)}_{1, 2} & W^{(1)}_{1, 3}
# \end{array}
# \right]
# \left(
# \begin{array}{c}
# a^{(1)}_1 \\
# a^{(1)}_2 \\
# a^{(1)}_3
# \end{array}
# \right)
# +
# b^{(1)}_1 \in \R.
# \end{aligned}
# $$
#
# This is done for each entry in the data set. Therefore, if the data set is of size $(n, 2)$, each variable that appears in the forward pass will be of size
#
# $$
# \begin{aligned}
# &\boldsymbol{a}^{(0)}: (2, n), \\
# &\boldsymbol{z}^{(0)}: (3, n), \\
# &\boldsymbol{a}^{(1)}: (3, n), \\
# &\boldsymbol{a}^{(2)}: (1, n).
# \end{aligned}
# $$


# %% [markdown]
# ## 1.1 Forward pass

# %% [markdown]
# We recall the forward pass implementation for shallow networks from PC2 (Exercise 2).


# %%
def forward_pass_shallow(X, weights, biases, activation=sigmoid()):
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
    Z, A = [], [X.T]

    z = weights[0] @ A[0] + biases[0]
    a = activation(z)
    Z.append(z)
    A.append(a)

    A.append(weights[1] @ A[1] + biases[1])

    return Z, A


# %% [markdown]
# ## 1.2 Backpropagation

# %% [markdown]
# The backward pass in our example for a given $y\in\R$ can be written as follows:
#
# $$
# \begin{aligned}
# &\delta b^{(1)} = \delta z^{(1)} = 2(a^{(2)} - y) \in \R, \\
# &\delta \boldsymbol{W}^{(1)} = \delta z^{(1)} (\boldsymbol{a}^{(1)})^T \in \R^{1\times 3}, \\
# &\delta \boldsymbol{b}^{(0)} = \delta \boldsymbol{z}^{(0)} = \sigma'(\boldsymbol{z}^{(0)})(\boldsymbol{W}^{(1)})^T \delta z^{(1)} \in \R^3, \\
# &\delta \boldsymbol{W}^{(0)} = \delta \boldsymbol{z}^{(0)} (\boldsymbol{a}^{(0)})^T \in \R^{3\times 2}.
# \end{aligned}
# $$
#
# This is also done for each entry in the data set. An extra $1/n$ factor will also appear.

# %% [markdown]
# ## Exercise
#
# Given the following implementation of backward propagation (see Theorem 1.10), fill in the blanks and justify the $1/n$ factor.


# %%
def backpropagation_shallow(X, y, weights, biases, activation=sigmoid()):
    """
    Performs backpropagation to compute the gradients of the loss function with respect to weights and biases.

    Parameters
    ----------
    X: numpy.ndarray
        Input data of shape (n, d)
    y: numpy.ndarray
        True labels of shape (n).
    weights: list of numpy.ndarray
        List of weight matrices for each layer.
    biases: list of numpy.ndarray
        List of bias vectors for each layer.
    activation: callable, optional
        Activation function to use. Default is sigmoid().

    Returns
    -------
    dW: list of numpy.ndarray
        Gradients of the loss with respect to weights for each layer.
    db: list of numpy.ndarray
        Gradients of the loss with respect to biases for each layer.
    """
    n = y.shape[0]

    Z, A = forward_pass_shallow(X, weights, biases, activation)

    dW = [np.empty_like(w) for w in weights]
    db = [np.empty_like(b) for b in biases]

    dZ = 2 * (A[2] - y.T)
    db[1] = np.sum(dZ, axis=1, keepdims=True) / n
    dW[1] = dZ @ A[1].T / n

    dZ = None  # To implement
    db[0] = None  # To implement
    dW[0] = None  # To implement

    return dW, db


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

# Backpropagation:
dW, db = backpropagation_shallow(X, y, weights, biases)
print(len(dW), len(db))
print(dW)
print(db)
print(db[1].shape)
print(dW[1].shape)
print(db[0].shape)
print(dW[0].shape)


# %% [markdown]
# ## 1.3 Gradient descent

# %% [markdown]
# We provide the following function to compute the MSE.


# %%
def MSE_shallow(X, y, weights, biases, activation=sigmoid()):
    """
    Computes the Mean Squared Error (MSE) between the predicted outputs and the actual targets.

    Parameters
    ----------
    X: numpy.ndarray
        Input data of shape (n, d).
    y: numpy.ndarray
        Actual target values of shape (n).
    weights: list of numpy.ndarray
        List of weight matrices for each layer in the neural network.
    biases: list of numpy.ndarray
        List of bias vectors for each layer in the neural network
    activation: callable, optional
        Activation function to use. Default is sigmoid().

    Returns
    -------
    float
        The mean squared error between the predicted outputs and the actual targets.
    """
    _, A = forward_pass_shallow(X, weights, biases, activation)
    return np.square(A[-1] - y.T).mean()


# %% [markdown]
# ## Exercise
#
# Given the following implementation of gradient descent (see Theorem 1.5), fill in the blanks.


# %%
def gradient_descent(
    X,
    y,
    XX,
    yy,
    weights,
    biases,
    *,
    activation=sigmoid(),
    learning_rate=0.01,
    epoch=1000,
):
    """
    Performs gradient descent to minimize the mean squared error (MSE).

    Parameters
    ----------
    X: numpy.ndarray
        Input features matrix (training set).
    y: numpy.ndarray
        Target values vector (training set).
    XX: numpy.ndarray
        Input features matrix (testing set).
    yy: numpy.ndarray
        Target values vector (testing set).
    weights: list of numpy.ndarray
        List of weight matrices for each layer.
    biases: list of numpy.ndarray
        List of bias vectors for each layer.
    activation: callable, optional
        Activation function to use. Default is sigmoid().
    learning_rate: float, optional
        Learning rate for gradient descent updates. Default is 0.01.
    epoch: int, optional
        Number of iterations for gradient descent. Default is 1000.

    Returns
    -------
    error_train: numpy.ndarray
        Array of MSE values for each epoch (training set).
    error_test: numpy.ndarray
        Array of MSE values for each epoch (testing set).
    """
    error_train = np.empty(epoch + 1)
    error_train[0] = None  # To implement
    error_test = np.empty(epoch + 1)
    error_test[0] = None  # To implement
    ie = 1

    for _ in range(epoch):
        dW, db = None  # To implement

        for i in range(len(biases)):
            biases[i] -= None  # To implement
            weights[i] -= None  # To implement

        error_train[ie] = None  # To implement
        error_test[ie] = None  # To implement
        ie += 1

    return error_train, error_test


# %% [markdown]
# ### Validation

# %%
# Setup (data):
d = 5
n = 20 * d
eta = 1e-2
data = "sphere"
X, y = get_dataset(data, n, d, eta)
XX, yy = get_dataset(data, n, d, eta, seed=43)

# Setup (NN):
d1 = 2 * d
L = 1

# Setup (optim):
lr = 2e-1
ep = 20000

# Initialize:
weights, biases = init(d, d1, L)

# Optimize:
t0 = time.time()
error_train, error_test = gradient_descent(
    X, y, XX, yy, weights, biases, learning_rate=lr, epoch=ep
)
t1 = time.time()
print(f"Time:           {t1-t0:.2f}s")

# Compute prediction and errors:
_, A = forward_pass_shallow(X, weights, biases)
_, AA = forward_pass_shallow(XX, weights, biases)
print(f"MSE (training): {np.square(A[-1][0] - y).mean():.2e}")
print(f"MSE (testing):  {np.square(AA[-1][0] - yy).mean():.2e}")

# Plot:
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].loglog(error_train, "-", label="MSE (training)")
axs[0].loglog(error_test, "-", label="MSE (testing)")
axs[0].legend()
n_disp = min(n, 50)
axs[1].plot(A[-1][0][:n_disp], "-", color="tab:blue", label="prediction (training)")
axs[1].plot(y[:n_disp], "--k", label="exact (training)")
axs[1].legend()
axs[2].plot(AA[-1][0][:n_disp], "-", color="tab:orange", label="prediction (testing)")
axs[2].plot(yy[:n_disp], "--k", label="exact (testing)")
axs[2].legend()

# %% [markdown]
# # 2 Deep networks
#
# In this section, we modify the previous functions to be able
#
# - to change the number of hidden layers;
# - to implement stochastic gradient descent;
# - to shuffle the data if needed.


# %% [markdown]
# ## 2.1 Forward pass

# %% [markdown]
# We recall the forward pass implementation for deep networks from PC2.


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
        z = weights[l - 1] @ A[l - 1] + biases[l - 1]
        a = activation(z)
        Z.append(z)
        A.append(a)

    A.append(weights[L] @ A[L] + biases[L])

    return Z, A


# %% [markdown]
# We also provide the following function.


# %%
def MSE_deep(X, y, weights, biases, activation=sigmoid()):
    """
    Computes the Mean Squared Error (MSE) between the predicted outputs and the actual targets.

    Parameters
    ----------
    X: numpy.ndarray
        Input data of shape (n, d).
    y: numpy.ndarray
        Actual target values of shape (n).
    weights: list of numpy.ndarray
        List of weight matrices for each layer in the neural network.
    biases: list of numpy.ndarray
        List of bias vectors for each layer in the neural network
    activation: callable, optional
        Activation function to use. Default is sigmoid().

    Returns
    -------
    float
        The mean squared error between the predicted outputs and the actual targets.
    """
    _, A = forward_pass_deep(X, weights, biases, activation)
    return np.square(A[-1] - y.T).mean()


# %% [markdown]
# ## 2.2 Backpropagation

# %% [markdown]
# ## Exercise
#
# Given the following implementation of backpropagation (see Theorem 1.10), fill in the blanks.


# %%
def backpropagation_deep(X, y, weights, biases, activation=sigmoid()):
    """
    Performs backpropagation to compute the gradients of the loss function with respect to weights and biases.

    Parameters
    ----------
    X: numpy.ndarray
        Input data of shape (n, d)
    y: numpy.ndarray
        True labels of shape (n).
    weights: list of numpy.ndarray
        List of weight matrices for each layer.
    biases: list of numpy.ndarray
        List of bias vectors for each layer.
    activation: callable, optional
        Activation function to use. Default is sigmoid().

    Returns
    -------
    dW: list of numpy.ndarray
        Gradients of the loss with respect to weights for each layer.
    db: list of numpy.ndarray
        Gradients of the loss with respect to biases for each layer.
    """
    L = len(weights) - 1
    n = y.shape[0]

    Z, A = None  # To implement

    dW = [np.empty_like(w) for w in weights]
    db = [np.empty_like(b) for b in biases]

    dZ = 2 * (A[L + 1] - y.T)
    db[L] = np.sum(dZ, axis=1, keepdims=True) / n
    dW[L] = dZ @ A[L].T / n

    for l in range(L, 0, -1):
        dZ = None  # To implement
        db[l - 1] = None  # To implement
        dW[l - 1] = None  # To implement

    return dW, db


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

# Backpropagation:
dW, db = backpropagation_deep(X, y, weights, biases)
print(len(dW), len(db))
print(dW)
print(db)


# %% [markdown]
# ## 2.3 Stochastic gradient descent

# %% [markdown]
# - In Gradient Descent (GD), at each iteration (or epoch), the gradient is computed using the entire dataset.
#
# - In Stochastic Gradient Descent (SGD), the gradient is computed on a smaller subset of the dataset. During the lecture, we examined the simplest version of this approach, where only a single data point is drawn randomly. Here, we will explore additional strategies that are commonly used in practice.
#
# For all strategies, we define a *batch size*. For example, if the dataset contains $n=8$ data points and the batch size is $2$, we will consider $4$ chunks of size $2$. The difference between strategies lies in how the chunks are selected. Note that a chunk is defined by indices, which correspond to the rows of the data matrix.
#
# - Strategy 1: The indices $[0,1,2,3,4,5,6,7]$ are split sequentially into $[0,1]$, $[2,3]$, $[4,5]$, and $[6,7]$;
#
# - Strategy 2: The indices are first randomly permuted, e.g., $[4,2,5,7,6,1,3,0]$, and then split into $[4,2]$, $[5,7]$, $[6,1]$, and $[3,0]$.
#
# - Strategy 3: $8$ indices are drawn randomly, e.g., $[4,0,1,7,6,2,6,0]$, and then split into $[4,0]$, $[1,7]$, $[6,2]$, and $[6,0]$.
#
# Strategy 1 will correspond to `shuffle=None`, Strategy 2 to `shuffle=permutation`, and Strategy 3 to `shuffle=randint`. (Note that Strategy 3 with a batch size $1$ corresponds to what we covered in the lecture.)
#
# For these strategies, an epoch is defined as one complete pass through all the chunks.
#
# Finally, we also added the Adam algorithm (see https://arxiv.org/abs/1412.6980).

# %% [markdown]
# ## Exercise
#
# Given the following implementation of stochastic gradient descent (see Theorem 1.8), fill in the blanks.


# %%
def stochastic_gradient_descent(
    X,
    y,
    XX,
    yy,
    weights,
    biases,
    *,
    activation=sigmoid(),
    learning_rate=0.01,
    epoch=1000,
    optimizer="SGD",
    batch_size=None,
    shuffle=None,
):
    """
    Performs gradient descent to minimize the mean squared error (MSE).

    Parameters
    ----------
    X: numpy.ndarray
        Input features matrix (training set).
    y: numpy.ndarray
        Target values vector (training set).
    XX: numpy.ndarray
        Input features matrix (testing set).
    yy: numpy.ndarray
        Target values vector (testing set).
    weights: list of numpy.ndarray
        List of weight matrices for each layer.
    biases: list of numpy.ndarray
        List of bias vectors for each layer.
    activation: callable, optional
        Activation function to use. Default is sigmoid().
    learning_rate: float, optional
        Learning rate for gradient descent updates. Default is 0.01.
    epoch: int, optional
        Number of iterations for gradient descent. Default is 1000.
    optimizer: string, optional
        Optimizer used to update the weights and biases. Default is 'SGD'. 'Adam' is the other choice.
    batch_size: int, optional
        Size of the mini-batches. If None, use the entire dataset. Default is None.
    shuffle: string, optional
        Whether to shuffle the data before each epoch. Default is None. Possible values: 'permutation' or 'randint'.

    Returns
    -------
    error_train: numpy.ndarray
        Array of MSE values for each epoch (training set).
    error_test: numpy.ndarray
        Array of MSE values for each epoch (testing set).
    """
    if batch_size is None:
        batch_size = y.shape[0]

    nb_chunks = y.shape[0] // batch_size

    error_training = np.empty(nb_chunks * epoch + 1)
    error_training[0] = None  # To implement
    error_testing = np.empty(nb_chunks * epoch + 1)
    error_testing[0] = None  # To implement
    ie = 1

    if optimizer == "Adam":
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        m_t_w = [np.zeros_like(w) for w in weights]
        m_t_b = [np.zeros_like(b) for b in biases]
        v_t_w = [np.zeros_like(w) for w in weights]
        v_t_b = [np.zeros_like(b) for b in biases]

    t = 0

    for iepoch in range(epoch):
        if shuffle == "permutation":
            indices = None  # To implement
        elif shuffle == "randint":
            indices = None  # To implement
        else:
            indices = None  # To implement

        for X_, y_ in zip(
            np.array_split(X[indices, :], nb_chunks),
            np.array_split(y[indices], nb_chunks),
        ):
            dW, db = backpropagation_deep(X_, y_, weights, biases, activation)

            if optimizer == "SGD":
                for i in range(len(biases)):
                    biases[i] -= None  # To implement
                    weights[i] -= None  # To implement

            if optimizer == "Adam":
                for i in range(len(biases)):
                    m_t_b[i] = beta1 * m_t_b[i] + (1 - beta1) * db[i]
                    v_t_b[i] = beta2 * v_t_b[i] + (1 - beta2) * (db[i] * db[i])

                    m_t_w[i] = beta1 * m_t_w[i] + (1 - beta1) * dW[i]
                    v_t_w[i] = beta2 * v_t_w[i] + (1 - beta2) * (dW[i] * dW[i])
                t += 1

                for i in range(len(biases)):
                    m_hat = m_t_b[i] / (1 - beta1**t)
                    v_hat = v_t_b[i] / (1 - beta2**t)
                    biases[i] -= None  # To implement

                    m_hat = m_t_w[i] / (1 - beta1**t)
                    v_hat = v_t_w[i] / (1 - beta2**t)
                    weights[i] -= None  # To implement

            error_training[ie] = None  # To implement
            error_testing[ie] = None  # To implement
            ie += 1

    return error_training, error_testing


# %% [markdown]
# ### Validation

# %% editable=true slideshow={"slide_type": ""}
# Setup (data):
d = 5
n = 20 * d
eta = 1e-2
data = "sphere"
X, y = get_dataset(data, n, d, eta)
XX, yy = get_dataset(data, n, d, eta, seed=43)

# Setup (NN):
d1 = 2 * d
L = 2
act = ReLU()

# Setup (optim):
lr = 1e-2
bs = n // 10
ep = 5000 * bs // n
op = "Adam"
sf = "permutation"

# Initialize:
weights, biases = init(d, d1, L)

# Optimize:
t0 = time.time()
error_train, error_test = stochastic_gradient_descent(
    X,
    y,
    XX,
    yy,
    weights,
    biases,
    optimizer=op,
    shuffle=sf,
    batch_size=bs,
    learning_rate=lr,
    epoch=ep,
    activation=act,
)
t1 = time.time()
print(f"Time:           {t1-t0:.2f}s")

# Compute prediction and errors:
_, A = forward_pass_deep(X, weights, biases, act)
_, AA = forward_pass_deep(XX, weights, biases, act)
print(f"MSE (training): {np.square(A[-1][0] - y).mean():.2e}")
print(f"MSE (testing):  {np.square(AA[-1][0] - yy).mean():.2e}")

# Plot:
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].loglog(error_train[0 :: n // bs], "-", label="MSE (training)")
axs[0].loglog(error_test[0 :: n // bs], "-", label="MSE (testing)")
axs[0].legend()
n_disp = min(n, 50)
axs[1].plot(A[-1][0][:n_disp], "-", color="tab:blue", label="prediction (training)")
axs[1].plot(y[:n_disp], "--k", label="exact (training)")
axs[1].legend()
axs[2].plot(AA[-1][0][:n_disp], "-", color="tab:orange", label="prediction (testing)")
axs[2].plot(yy[:n_disp], "--k", label="exact (testing)")
axs[2].legend()

# %% [markdown]
# ## 2.4 The influence of the number of neurons

# %% [markdown]
# Here, we look at the influence of the number of neurons $d_1$.

# %%
# Setup (data):
d = 5
n = 20 * d
eta = 1e-2
data = "sphere"
X, y = get_dataset(data, n, d, eta)
XX, yy = get_dataset(data, n, d, eta, seed=43)

# Setup (NN):
L = 2
act = ReLU()

# Setup (optim):
lr = 1e-2
bs = n // 10
ep = 5000 * bs // n
op = "Adam"
sf = "permutation"

# Loop:
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for d1 in np.arange(d // 2, 6 * d // 2, d // 2):
    weights, biases = init(X.shape[1], d1, L)
    t0 = time.time()
    error_train, error_test = stochastic_gradient_descent(
        X,
        y,
        XX,
        yy,
        weights,
        biases,
        optimizer=op,
        shuffle=sf,
        batch_size=bs,
        learning_rate=lr,
        epoch=ep,
        activation=act,
    )
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    axs[0].loglog(error_train[0 :: n // bs], label=f"{d1} neurons")
    axs[1].loglog(error_test[0 :: n // bs], label=f"{d1} neurons")
axs[0].set_title("Training set")
axs[0].set_xlabel("epochs")
axs[0].set_ylabel("MSE")
axs[0].legend()
axs[1].set_title("Testing set")
axs[1].set_xlabel("epochs")
axs[1].set_ylabel("MSE")
axs[1].legend()

# %% [markdown]
# ## Exercise
#
# Try different datasets and comment on what you observe.

# %% [markdown]
# ## 2.5 The influence of the number of hidden layers

# %% [markdown]
# ## Exercise
#
# Set up a numerical experiement to illustrate the influence of the number of layers $L$ and comment on what you observe.

# %% [markdown]
# ## 2.6 The influence of the dataset size

# %% [markdown]
# ## Exercise
#
# Set up a numerical experiement to illustrate the influence of the dataset size $n$ and comment on what you observe.

# %% [markdown]
# ## 2.7 The influence of the batch size

# %% [markdown]
# ## Exercise
#
# Set up a numerical experiement to illustrate the influence of the batch size and comment on what you observe.
