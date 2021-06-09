from dc.utils.ImportantVars import LENGTH, DIMS
import numpy as np
from pathlib import Path
import rasterio as rio
import torch
from torch import nn


# Model to calculate y = Ax + B where x is an ensemble of estimates and A is a matrix of size
# (ensemble_members, num_pixels). The model then weights each pixel/ensemble member to get the best results.
class Regressor(nn.module):

    def __init__(self, num_ensemble=10, num_pixels=LENGTH, mx_lead=12):
        self.A = torch.randn((1, num_ensemble, num_pixels, mx_lead), requires_grad=True)
        self.b = torch.randn((1, num_pixels, mx_lead), requires_grad=True)

    def forward(self, x, pixel, lead_time):

        A = self.A[:, :, pixel, lead_time]
        b = self.b[:, pixel, lead_time]

        return A.mm(x) + b

def make_ensemble(ens_path: Path, day: str):

    f_list = [str(x) for x in ens_path.rglob(day+'*None.tif')]
    arrs = np.array([rio.open(x).read(list(range(1, 13))) for x in f_list])

    return arrs


def run_regression(ens_dir, target_dir, epochs, batch_size):

    loader = None
    train_loader, test_loader = None, None

    model = Regressor(10, LENGTH)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        for lead_time in range(12):
            for
        optimizer.zero_grad()
        x = None # get_data func
        y_true = None # loader get_data func
        y_hat = model(x)

        loss = criterion(y_hat, y_true)
        loss.backward()
        optimizer.step()


