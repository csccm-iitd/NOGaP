import sys
sys.path.append('/home/user/Documents/NOGaP/general/')
import gpytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from utilities3 import *
from pytorch_wavelets import DWT, IDWT, DWT1D, IDWT1D
from gpytorch.means import MultitaskMean

'''   
WNO1D - USED AS MEAN FUNCTION

'''

""" Def: 1d Wavelet layer """
device = 'cuda:1'
class WaveConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, level, dummy):
        super(WaveConv1d, self).__init__()

        """
        1D Wavelet layer. It does Wavelet Transform and Inverse Wavelet Transform.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level  
        self.dwt_ = DWT1D(wave='db8', J=self.level, mode='symmetric').to(dummy.device)
        self.mode_data, _ = self.dwt_(dummy)
        self.modes1 = self.mode_data.shape[-1]
        
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))

    # Convolution
    def mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute single tree Discrete Wavelet coefficients using some wavelet 
        # dwt = DWT1D(wave='db8', J=self.level, mode='symmetric').cuda()
        dwt = DWT1D(wave='db8', J=self.level, mode='symmetric').to(device)
        x_ft, x_coeff = dwt(x)
        
        # Multiply the final low pass and high pass coefficients
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[-1],  device=x.device)
        out_ft = self.mul1d(x_ft, self.weights1)
        x_coeff[-1] = self.mul1d(x_coeff[-1], self.weights1)
        
        # Reconstruct the signal
        # idwt = IDWT1D(wave='db8', mode='symmetric').cuda()
        idwt = IDWT1D(wave='db8', mode='symmetric').to(device)
        x = idwt((out_ft, x_coeff))
        return x

""" The forward operation """
class WNO1d(nn.Module):
    def __init__(self, width, level, dummy_data):
        super(WNO1d, self).__init__()

        """
        The WNO network. It contains 4 layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. 4 layers of the integral operators v(+1) = g(K(.) + W)(v).
            W is defined by self.w_; K is defined by self.conv_.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.level = level
        self.width = width
        self.dummy_data = dummy_data
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        self.conv0 = WaveConv1d(self.width, self.width, self.level, self.dummy_data)
        self.conv1 = WaveConv1d(self.width, self.width, self.level, self.dummy_data)
        self.conv2 = WaveConv1d(self.width, self.width, self.level, self.dummy_data)
        self.conv3 = WaveConv1d(self.width, self.width, self.level, self.dummy_data)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # print(f"1 x : {x.shape}")
        x = x.reshape(x.shape[0],40,-1).float()
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # do padding, if required

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # remove padding, when required
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        # print(f"8 x : {x.shape}")
        return x

    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

#%%
'''   
Wrapping the mean wno2d using a class CustomMean()
'''

class CustomMean(gpytorch.means.Mean):
    def __init__(self, width, level, dummy_data ):
        super().__init__()
        self.wno = WNO1d(width, level, dummy_data)
        

    def forward(self, x):
        # x: Input data tensor of shape (batch_size, num_inputs)
                
        # Pass the reshaped input through the WNO1d function
        mean_prediction = self.wno(x)

        # return mean_prediction.reshape(mean_prediction.shape[0]* mean_prediction.shape[0])
        return mean_prediction


# FOR MULTIOUPUT GP- CUSTOM MULTITASK MEAN

class CustomMultitaskMean(MultitaskMean):
    def __init__(self, custom_mean, num_tasks):
        super().__init__(base_means=[gpytorch.means.ConstantMean()], num_tasks=num_tasks)
        self.custom_mean = custom_mean
        

    def forward(self, input):
        mean_prediction = self.custom_mean(input)
        return mean_prediction

