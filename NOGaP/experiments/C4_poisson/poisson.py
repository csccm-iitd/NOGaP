'''   


'''
#%%
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'general')))
import sys
sys.path.append('/home/user/Documents/NOGaP/general/')
import gpytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from pytorch_wavelets import DWT, IDWT
from gpytorch.means import MultitaskMean
from utilities3 import *
from mean_func import *
from utils import *
import scipy

# Constants and Paths
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_PATH = '/home/user/Documents/GP_WNO/DATA/u_sol_poissons.mat'
NTRAIN = 520
NTEST = 50
STEP_SIZE = 50
GAMMA = 0.75
LEVEL = 4
WIDTH = 64
R = 2
H = int(((66 - 1)/R) + 1)
S = H
ITERATIONS = 1500
LR = 0.01
print(f"Using device: {DEVICE}")

torch.manual_seed(0)
np.random.seed(0)


class GaussianProcessRegression(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, width, level, dummy_data):
        super().__init__(train_x, train_y, likelihood)
        num_dims = train_y.shape[-1]

        self.mean_module = CustomMultitaskMean(
            CustomMean(width, level, dummy_data), num_tasks=num_dims
        )

        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.keops.MaternKernel(nu=2.5, ard_num_dims=num_dims, has_lengthscale=True), 
            num_tasks=num_dims, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


def load_and_preprocess_data(path, ntrain, ntest, s, r, device):
    reader = scipy.io.loadmat(path)
    x = torch.tensor(reader['x1d'], dtype=torch.float)
    y = torch.tensor(reader['y1d'], dtype=torch.float)
    usol = torch.tensor(reader['sol'], dtype=torch.float)

    x_train1 = torch.tensor(reader['mat_sd'], dtype=torch.float)
    y_train1 = usol

    x_train = x_train1[:ntrain, ::r, ::r][:, :s, :s]
    y_train = y_train1[:ntrain, ::r, ::r][:, :s, :s]

    x_test = x_train1[-ntest:, ::r, ::r][:, :s, :s]
    y_test = y_train1[-ntest:, ::r, ::r][:, :s, :s]

    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)

    x_train = x_train.reshape(ntrain, s, s, 1)
    x_test = x_test.reshape(ntest, s, s, 1)

    x_tr = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2], x_train.shape[3]).squeeze(-1)
    y_tr = y_train.reshape(y_train.shape[0], y_train.shape[1] * y_train.shape[2])

    x_t = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2], x_test.shape[3]).squeeze(-1)
    y_t = y_test.reshape(y_test.shape[0], y_test.shape[1] * y_test.shape[2])

    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_t, y_t = x_t.to(device), y_t.to(device)

    return x_tr, y_tr, x_t, y_t, x_normalizer, y_normalizer



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

def predict_model(model, likelihood, x_t, y_normalizer, s):
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

    mean_pred = mean_pred.reshape(x_t.shape[0], s, s)
    mean_pred = y_normalizer.decode(mean_pred.detach().cpu())
    mean_pred = mean_pred.reshape(mean_pred.shape[0], mean_pred.shape[1] * mean_pred.shape[2])

    var_pred = var_pred.reshape(x_t.shape[0], s, s)
    var_pred = y_normalizer.decode(var_pred.detach().cpu())
    var_pred = var_pred.reshape(var_pred.shape[0], var_pred.shape[1] * var_pred.shape[2])
    var_pred = torch.abs(var_pred)

    return mean_pred, var_pred


x_tr, y_tr, x_t, y_t, x_normalizer, y_normalizer = load_and_preprocess_data(DATA_PATH, NTRAIN, NTEST, S, R, DEVICE)

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=y_tr.shape[-1])
model = GaussianProcessRegression(x_tr, y_tr, likelihood, WIDTH, LEVEL,dummy_data=x_tr.reshape(-1,S,S).unsqueeze(-1).permute(0, 3, 1, 2))

model = model.to(DEVICE)
likelihood = likelihood.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

losses_nll, losses_mse = train_model(model, likelihood, optimizer, mll, x_tr, y_tr, num_iterations=ITERATIONS)
mean_pred, var_pred = predict_model(model, likelihood, x_t, y_normalizer, S)

mse_loss = nn.MSELoss()
prediction_error = mse_loss(mean_pred.to(DEVICE), y_t)
relative_error = torch.mean(torch.linalg.norm(mean_pred.to(DEVICE) - y_t, axis=1) / torch.linalg.norm(y_t, axis=1))

print(f'MSE Testing error: {prediction_error.item()}')
print(f'Mean relative error: {100 * relative_error} %')

# Save results
save_results(mean_pred, var_pred, model, save_dir='results')

#%%