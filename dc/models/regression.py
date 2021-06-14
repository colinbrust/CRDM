from dc.utils.ImportantVars import LENGTH, DIMS
import numpy as np
from pathlib import Path
import rasterio as rio
import torch
from torch import nn


# Model to calculate y = Ax + B where x is an ensemble of estimates and A is a matrix of size
# (ensemble_members, num_pixels, mx_lead). The model then weights each pixel/ensemble member to get the best results.
class Regressor(nn.module):

    def __init__(self, num_ensemble=10, num_pixels=LENGTH, mx_lead=12):
        self.A = torch.randn((num_ensemble, num_pixels, mx_lead), requires_grad=True)
        self.b = torch.randn((num_pixels, mx_lead), requires_grad=True)

    def forward(self, x, lead_time):

        A = self.A[:, :, lead_time]
        b = self.b[:, lead_time]

        return A.bmm(x) + b


