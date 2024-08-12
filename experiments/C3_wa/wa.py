#%%
import os
import sys
sys.path.append('/home/user/Documents/NOGaP/general/')

import gpytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utilities3 import *
from pytorch_wavelets import DWT, IDWT, DWT1D, IDWT1D
from mean_func import CustomMean, CustomMultitaskMean
# Constants and Paths
Path = '/home/user/Documents/NOGaP/experiments/C3_wa'
save_dir = f"{Path}/model/"
train_data_path = '/home/user/Documents/GP_WNO/DATA/train_IC2.npz'
test_data_path = '/home/user/Documents/GP_WNO/DATA/test_IC2.npz'

# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Set seeds
torch.manual_seed(1)
np.random.seed(1)

# Model Parameters
ntrain = 1000
ntest = 50
s = 40
level = 3
width = 96
num_iterations = 1500
learning_rate = 0.005
weight_decay = 1e-6

class GaussianProcessRegression(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, width, level, dummy_data):
        super().__init__(train_x, train_y, likelihood)
        num_dims = train_y.shape[-1]

        self.mean_module = CustomMultitaskMean(
            CustomMean(width, level, dummy_data), num_tasks=num_dims
        )

        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.keops.MaternKernel(nu=2.5, ard_num_dims=num_dims, has_lengthscale=True), num_tasks=num_dims, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

def load_npy_data(train_path, test_path):
    data_train = np.load(train_path)
    data_test = np.load(test_path)

    x_train, t_train, u_train = data_train["x"], data_train["t"], data_train["u"]
    x_test, t_test, u_test = data_test["x"], data_test["t"], data_test["u"]
    
    return x_train, u_train, x_test, u_test

def preprocess_data(x_train, u_train, x_test, u_test, ntrain, ntest, s):
    x_data_train = torch.tensor(u_train[:, 0, :])
    y_data_train = torch.tensor(u_train[:, -2, :])

    x_data_test = torch.tensor(u_test[:, 0, :])
    y_data_test = torch.tensor(u_test[:, -2, :])

    x_train = x_data_train[:ntrain, :]
    y_tr = y_data_train[:ntrain, :]

    x_test = x_data_test[:ntest, :]
    y_t = y_data_test[:ntest, :]

    x_train = x_train.reshape(ntrain, s, 1)
    x_test = x_test.reshape(ntest, s, 1)

    x_tr = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    x_t = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

    return x_tr, y_tr, x_t, y_t


def train_model(model, likelihood, optimizer, mll, x_tr, y_tr, num_iterations):
    model.train()
    likelihood.train()
    losses_nll = []
    losses_mse = []
    mse_loss = nn.MSELoss()

    for i in range(num_iterations):
        optimizer.zero_grad()
        output = model(x_tr)
        loss_nll = -mll(output, y_tr)
        loss_mse = mse_loss(output.mean, y_tr)
        loss = -mll(output, y_tr)
        loss.backward()
        optimizer.step()

        losses_nll.append(loss_nll.item())
        losses_mse.append(loss_mse.item())

        print(f'Iter {i + 1}/{num_iterations} - NLL Loss: {loss_nll.item():.3f} - MSE Loss: {loss_mse.item():.3f}')
    
    return losses_nll, losses_mse

def predict(model, likelihood, x_t):
    model.eval()
    likelihood.eval()
    all_means = []
    all_variances = []

    with torch.no_grad():
        for i in range(x_t.shape[0]):
            x_i = x_t[i:i+1, :]
            observed_pred = model(x_i)
            mean_pred_i = observed_pred.mean
            var_pred_i = observed_pred.variance
            all_means.append(mean_pred_i.cpu().numpy())
            all_variances.append(var_pred_i.cpu().numpy())

    mean_pred = torch.tensor(np.array(all_means)).squeeze(1)
    var_pred = torch.tensor(np.array(all_variances)).squeeze(1)

    return mean_pred, var_pred

def plot_results(mean_pred, var_pred, y_t, s):
    x_axis = np.linspace(0, 1, s)
    plt.plot(x_axis, mean_pred.detach().cpu().numpy()[2, :], ':', linewidth=4)
    plt.plot(x_axis, y_t.detach().cpu().numpy()[2, :], linewidth=2)
    plt.xlabel("$X$", weight='bold')
    plt.ylabel("$u$", weight='bold')
    plt.xticks(weight='bold')
    plt.yticks(weight='bold')
    plt.show()

    plt.plot(var_pred.detach().cpu().numpy()[:, 27], label='var a/c N*_test')
    plt.xlabel("$X$", weight='bold')
    plt.ylabel("Var", weight='bold')
    plt.xticks(weight='bold')
    plt.yticks(weight='bold')
    plt.legend()
    plt.show()

    plt.plot(x_axis, var_pred.detach().cpu().numpy()[1, :], label='Spatial variance')
    plt.xlabel("$X$", weight='bold')
    plt.ylabel("Var", weight='bold')
    plt.xticks(weight='bold')
    plt.yticks(weight='bold')
    plt.legend()
    plt.show()


# Load and preprocess data
x_train, u_train, x_test, u_test = load_npy_data(train_data_path, test_data_path)
# x_train = torch.from_numpy(x_train)
x_tr, y_tr, x_t, y_t = preprocess_data(x_train, u_train, x_test, u_test, ntrain, ntest, s)

x_tr, y_tr = x_tr.to(device), y_tr.to(device)
x_t, y_t = x_t.to(device), y_t.to(device)
# x_train = torch.from_numpy(x_train)
# Instantiate the GP model
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=y_tr.shape[-1]).to(device)
model = GaussianProcessRegression(x_tr, y_tr, likelihood, width =width, level=level, dummy_data=x_tr.unsqueeze(-1).permute(0, 2, 1))


model = model.to(device)
likelihood = likelihood.to(device)

# Optimizer and MLL
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Train the model
losses_nll, losses_mse = train_model(model, likelihood, optimizer, mll, x_tr, y_tr, num_iterations)

# Predict
mean_pred, var_pred = predict(model, likelihood, x_t)

# Compute prediction errors
mse_loss = nn.MSELoss()
prediction_error = mse_loss(mean_pred.to(device), y_t)
relative_error = torch.mean(torch.linalg.norm(mean_pred.to(device) - y_t, axis=1) / torch.linalg.norm(y_t, axis=1))

print(f'MSE Testing error: {prediction_error.item()}')
print(f'Mean relative error: {100 * relative_error} %')

# Plot results
plot_results(mean_pred, var_pred, y_t, s)


#%%