from dc.utils.ImportantVars import LENGTH, DIMS
import numpy as np
from pathlib import Path
import rasterio as rio
import torch
from torch import nn


# Model to calculate y = Ax + B where x is an ensemble of estimates and A is a matrix of size
# (ensemble_members, num_pixels, mx_lead). The model then weights each pixel/ensemble member to get the best results.
class Regressor(nn.Module):

    def __init__(self, num_ensemble=10, num_pixels=LENGTH, mx_lead=12):
        super().__init__()    
    
        self.A = nn.Parameter(torch.randn((1, num_ensemble, num_pixels, mx_lead), requires_grad=True))
        self.b = nn.Parameter(torch.randn((1, num_pixels, mx_lead), requires_grad=True))

    def forward(self, x, lead_time):
        
        A = self.A[:, :, lead_time]
        b = self.b[:, lead_time]
        print(A.shape)
        print(x.shape)
        return A.mm(x) + b


