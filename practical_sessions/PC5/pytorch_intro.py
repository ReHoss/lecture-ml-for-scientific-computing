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
# <i>PyTorch introduction</i>
# </div>
# <div style="text-align: center;">
# </br>
# <p>Victorita Dolean, Loïc Gouarin, Rémy Hosseinkhan, Hadrien Montanelli
# </br>
# 2025-2026
# </p>
# </div>

# %% [markdown]
# # 1 Introduction


# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import time

# %% [markdown]
# During the first practical sessions, you wrote on your own different algorithms that are the angular key of solving problem using neural networks. In the rest of this course, we will see more complex examples and you will not implement all the methods that we will use. You have now enough background to understand how it works under the hood.
#
# This notebook is devoted to introducing you to the neural network software that we will be using until the end of the course. You have several options, but in Python, the three most commonly used packages are TensorFlow, PyTorch, and Jax. We will choose PyTorch for the following.
#
# The first step is to import the PyTorch package.

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# %% [markdown]
# Let us set a seed for the pseudo-random number generator used in `numpy` and `pytorch` for the sake of results reproductibility.

# %%
np.random.seed(42)
torch.manual_seed(42)


# %% [markdown]
# We will try to create a neural network similar to the one used in the introduction of the PC 2. We recall that we had the possibility to create a dataset using the following function.


# %%
def get_dataset(datatype, n, d, eta, *, random_gen=np.random.rand):
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


# %% [markdown]
# # 2 Definition of a model using PyTorch nn.Sequential

# %% [markdown]
# PyTorch proposes routines and data structures called [Sequential model](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html), which enable to specify the model layer by layer.
#
# Let us implement a first Neural Network (NN) model $\widehat{f}$ defined by:
# * a single hidden layer ($L=1$);
# * an input in dimension $100$ ($d_0=d=100$);
# * the first hidden layer with $128$ neurons ($d_1=128$);
# * a scalar output layer ($d_2=1$).
#
# The function $\widehat{f}$ can therefore be written as
#
# $$
# \widehat{f}(\cdot;\bm{w})
# =
# a^{(2)}
# \circ
# \bm{a}^{(1)}
# \circ
# \bm{a}^{(0)}
# :\R^{100}\to\R
# $$
#
# \begin{align}
# & \bm{a}^{(0)}:\R^{100}\to\R^{100}, && \bm{a}^{(0)}(\bm{x}) = \bm{x}, \\
# & \bm{a}^{(1)}:\R^{100}\to\R^{128}, && \bm{a}^{(1)}(\bm{x}) = \sigma\left(W^{(1)}\bm{x} + \bm{b}^{(1)}\right), \\
# & {a}^{(2)}:\R^{128}\to\R, && {a}^{(2)}(\bm{x}) = W^{(2)}\bm{x} + b^{(2)}.
# \end{align}
#

# %% [markdown]
# First define an empty model.

# %%
model = nn.Sequential()

# %% [markdown]
# ## 2.1 Adding layers

# %% [markdown]
# In our case the structure of the model is:
# 1. an input of size ($d_0=d=100$);
# 2. a single hidden layer ($L=1$) with 128 neurons ($d_1=128$);
# 3. an output of size 1 ($d_2=1$).

# %% [markdown]
# The layers can be added to the model sequentially using the function
# `nn.Linear()`.

# %%
model = nn.Sequential(nn.Linear(100, 128), nn.Linear(128, 1))

# %%
print(model)

# %% [markdown]
# By default, PyTorch layers don't include activation functions. Therefore the above python lines create linear layers only.
#
# We can specify an activation function for each layer when it is added into the model.
#
# Let us implement the previous model with a **sigmoid activation function** in the hidden layer. PyTorch provides an implementation of the sigmoid function in `torch.nn.Sigmoid()`.

# %%
model = nn.Sequential(nn.Linear(100, 128), nn.Sigmoid(), nn.Linear(128, 1))

# %% [markdown]
# Other activation functions can be found in `torch.nn`: the complete list of [PyTorch activation functions](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) is available in the official documentation.

# %% [markdown]
# One can specify how the weight and biases are initialized by manually initializing the layers after model creation.

# %%
# setting the initializer
model = nn.Sequential(nn.Linear(100, 128), nn.Sigmoid(), nn.Linear(128, 1))

# Initialize weights uniformly
for layer in model:
    if isinstance(layer, nn.Linear):
        nn.init.uniform_(layer.weight, -0.1, 0.1)
        nn.init.uniform_(layer.bias, -0.1, 0.1)

# %% [markdown]
# The complete list of [initializers](https://pytorch.org/docs/stable/nn.init.html) is available in the official documentation. By default, PyTorch uses [Kaiming uniform initialization](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_) for weights.

# %% [markdown]
# We can inspect the characteristics of each layer as follows.

# %%
for name, layer in model.named_modules():
    if isinstance(layer, nn.Linear):
        print(f"Layer: {name}, Type: {type(layer).__name__}")
    elif isinstance(layer, (nn.Sigmoid, nn.ReLU, nn.Tanh)):
        print(f"Activation: {name}, Type: {type(layer).__name__}")

# %% [markdown]
# ## 2.2 Specifying an optimization algorithm and a loss function

# %% [markdown]
# Now that we have implemented the model, we can specify the loss function and the algorithm that will be used to optimize its parameters.

# %% [markdown]
# We choose the Mean Square Error (MSE) as loss function in order to fit our model to the data. PyTorch provides an implementation of several loss functions.

# %%
loss_fn = nn.MSELoss()

# %% [markdown]
# Let us now choose the Stochastic Gradient Descent (SGD) as the optimization algorithm and specify its learning rate.

# %%
optimizer = optim.SGD(model.parameters(), lr=1e-2)

# %% [markdown]
# PyTorch provides other loss functions in `torch.nn` and the optimizers in `torch.optim`.
#
# The complete list of [PyTorch optimizers](https://pytorch.org/docs/stable/optim.html) is available in the official documentation.
#
# The complete list of [PyTorch loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions) is available in the official documentation.

# %% [markdown]
# # 3 Training the model

# %% [markdown]
# We now turn to the training of the model.
#
# Let us first create a training and a testing data set, define a model, a loss function and an optimizer.

# %%
# Setup (data):
d = 5
n = 20 * d
eta = 1e-2
data = "sphere"
X, y = get_dataset(data, n, d, eta)
XX, yy = get_dataset(data, n, d, eta)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X)
y_train = torch.FloatTensor(y).reshape(-1, 1)
X_test = torch.FloatTensor(XX)
y_test = torch.FloatTensor(yy).reshape(-1, 1)

# Setup (NN):
d1 = 2 * d
model = nn.Sequential(nn.Linear(d, d1), nn.Sigmoid(), nn.Linear(d1, 1))

# Setup (optim):
lr = 2e-1
ep = 250
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
print(model)

# %% [markdown]
# In PyTorch, we need to implement the training loop manually. This gives us more flexibility but requires more code.
# We'll create a training function that also evaluates the model against a testing dataset during the training.
#
# The function returns dictionaries that gather information recorded during the training process like the evolution of the loss
# against the training data and the testing data.


# %%
def train_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    loss_fn,
    optimizer,
    epochs,
    batch_size,
    shuffle=True,
):
    """
    Train a PyTorch model.

    Parameters:
    - model: the neural network model
    - X_train, y_train: training data
    - X_test, y_test: testing data
    - loss_fn: loss function
    - optimizer: optimizer
    - epochs: number of epochs
    - batch_size: batch size
    - shuffle: whether to shuffle data

    Returns:
    - history: dictionary with 'loss' and 'val_loss' lists
    """
    history = {"loss": [], "val_loss": []}

    # Create data loader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)

        epoch_loss /= len(X_train)
        history["loss"].append(epoch_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = loss_fn(val_outputs, y_test)
            history["val_loss"].append(val_loss.item())

    return history


# %%
t0 = time.time()
history = train_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    loss_fn,
    optimizer,
    ep,
    batch_size=25,
    shuffle=True,
)
t1 = time.time()
print(f"Time: {t1-t0:.2f}s")

# %% [markdown]
# Thanks to the output of the training function, we can plot the evolution of the loss.

# %%
plt.loglog(history["loss"], label="MSE (training)")
plt.loglog(history["val_loss"], label="MSE (testing)")
plt.grid()
plt.xlabel("epochs")
plt.ylabel("MSE")
plt.legend()


# %% [markdown]
# The training function has similar options to what we have implemented in the previous practical session:
#
# - `epochs` gives the number of iterations for the optimization algorithm;
# - `batch_size` is the number of data that extracted to update the weights and biases;
# - `shuffle` is used to shuffle the data if needed.

# %% [markdown]
# # 4 Evaluating the model

# %% [markdown]
# The model can be called using `model()` against a data set for evaluation.

# %%
# Predictions:
model.eval()
with torch.no_grad():
    ypred = model(X_train).numpy()
    yypred = model(X_test).numpy()

# Convert to numpy for plotting
y_np = y_train.numpy()
yy_np = y_test.numpy()

# Plot:
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].plot(y_np, ypred, "o", label="prediction (training)")
axs[0].plot(yy_np, yypred, ".", label="prediction (testing data)")
axs[0].plot(yy_np, yy_np, "r-", label="expected result")
axs[0].legend()
n_disp = min(n, 50)
axs[1].plot(ypred, "-", color="tab:blue", label="prediction (training)")
axs[1].plot(y_np, "--k", label="exact (training)")
axs[1].legend()
axs[2].plot(yypred, "-", color="tab:orange", label="prediction (testing)")
axs[2].plot(yy_np, "--k", label="exact (testing)")
axs[2].legend()


# %% [markdown]
# # 5 Playing with a multilayer perceptron in PyTorch

# %% [markdown]
# Now that you have all the ingredients to create and train a neural network using PyTorch, you can try to modify the previous model by:
# - changing the number of hidden layers;
# - changing the number of neurons in each hidden layer;
# - changing the activation functions;
# - changing the optimization algorithm;
# - changing the learning rate;
# - changing the batch size;
# - changing the number of epochs.
#   

# %%
def create_model_adam(
    d: int,
    L: int,
    d1: int,
    *,
    activation="linear",
    lr: float = 1e-2,
    verbose: bool = False,
) -> nn.Module:
    """
    Create a neural network model with the given parameters.
    Parameters:
    - d: int, the size of the input layer.
    - L: int, the number of hidden layers.
    - d1: int, the number of neurons in each hidden layer.
    - activation: string, the activation function to use in the hidden layers
    - lr: float, the learning rate of the optimizer.
    - verbose: bool, if True, print a summary of the model.
    Returns:
    - model: nn.Module, the created model.
    - optimizer: the Adam optimizer for this model.
    """
    # Map activation names to PyTorch activation functions
    activation_map = {
        "linear": nn.Identity(),
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
    }

    if activation not in activation_map:
        raise ValueError(f"Activation {activation} not supported")

    # Define the model
    layers = []

    # Add input and hidden layers
    for i in range(L):
        if i == 0:
            layers.append(nn.Linear(d, d1))
        else:
            layers.append(nn.Linear(d1, d1))
        layers.append(activation_map[activation])

    # Add output layer
    layers.append(nn.Linear(d1, 1))

    model = nn.Sequential(*layers)

    # Initialize weights uniformly
    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.uniform_(layer.weight, -1.0, 1.0)
            nn.init.uniform_(layer.bias, -1.0, 1.0)

    # Create the Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if verbose:
        # Summary of the model
        print(model)

    return model, optimizer


# %% [markdown]
# ### Validation


# %%
# Setup (data):
d = 5
n = 20 * d
eta = 1e-2
data = "sphere"
X, y = get_dataset(data, n, d, eta)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X)
y_train = torch.FloatTensor(y).reshape(-1, 1)

# Setup (NN):
d1 = 2 * d
L = 2
lr = 1e-2
model, optimizer = create_model_adam(d, L, d1, activation="relu", lr=lr)

# Setup (optim):
bs = n // 10
print(f"Batch size: {bs}")
ep = 5000 * bs // n
print(f"Number of epochs: {ep}")

# Create dummy test data for validation
X_test = X_train
y_test = y_train

# Train:
t0 = time.time()
history = train_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    nn.MSELoss(),
    optimizer,
    ep,
    batch_size=bs,
    shuffle=True,
)
t1 = time.time()
print(f"Time: {t1-t0:.2f}s")
plt.loglog(history["loss"], label="MSE (training)")
plt.grid()
plt.xlabel("epochs")
plt.ylabel("MSE")
plt.legend()

# %%
