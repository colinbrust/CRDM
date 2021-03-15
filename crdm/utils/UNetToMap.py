import argparse
import torch
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from crdm.classification.UNet import UNet
from crdm.loaders.ReadUNet import AggregateAllSpatial
from crdm.utils.ImportantVars import DIMS, LENGTH
from crdm.utils.ParseFileNames import parse_fname
import rasterio as rio

CROP_SIZE = 32


def make_pad(arr, crop_size):

    row_pad = int(np.ceil(DIMS[0]/crop_size) * crop_size - DIMS[0])
    col_pad = int(np.ceil(DIMS[1]/crop_size) * crop_size - DIMS[1])
    pad = ((0, 0), (row_pad, 0), (col_pad, 0))
    out = np.pad(arr, pad_width=pad, mode='constant', constant_values=-1.5)

    return row_pad, col_pad, out


def make_model(mod_f, cuda, n_channels):

    # make model from hyperparams and load trained parameters.
    model = UNet(n_channels=n_channels, n_classes=1)

    model.load_state_dict(torch.load(mod_f)) if cuda and torch.cuda.is_available() else model.load_state_dict(
        torch.load(mod_f, map_location=torch.device('cpu')))

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model


def get_pred_true_arrays(model, mod_f, target, in_features, lead_time, dt):

    # Get hyperparameters from filename
    info = parse_fname(mod_f)
    batch, nWeeks = int(info['batch']), int(info['nMonths'])
    data = AggregateAllSpatial(target=target, lead_time=lead_time, in_features=in_features, n_weeks=nWeeks)

    weeklys = data.premake_features()
    weeklys = weeklys.reshape(weeklys.shape[0], *DIMS)
    # Pad with no-data values so we get a clean image for a given crop size.
    row_pad, col_pad, weeklys = make_pad(weeklys, CROP_SIZE)

    nrow = weeklys.shape[1]
    ncol = weeklys.shape[2]

    col_indices = [x for x in range(0, ncol+1, CROP_SIZE)]
    row_indices = [x for x in range(0, nrow+1, CROP_SIZE)]

    arr_out = []

    for col in range(len(col_indices) - 1):    
        col_stack = []
        for row in range(len(row_indices) - 1):
            
            arr = weeklys[:, row_indices[row]:row_indices[row+1], col_indices[col]:col_indices[col+1]]
            arr = dt(arr)
            arr = torch.unsqueeze(arr, 0)
            
            out = model(arr)
            out = out.cpu().detach().numpy()
            out = out.squeeze()
            col_stack.append(out)

        col_stack = np.vstack(col_stack)
        arr_out.append(col_stack)

    out = np.hstack(arr_out)
    # Remove padding
    out = out[row_pad:, col_pad:]
    
    out = out.astype('float32')
    print(out.shape)
    return out


def save_arrays(out_dir, out, target):

    out_name = os.path.join(out_dir, os.path.basename(target).replace('_USDM.dat', '_preds.tif'))
    print(out_name)
    out_dst = rio.open(
        out_name,
        'w',
        driver='GTiff',
        height=DIMS[0],
        width=DIMS[1],
        count=1,
        dtype='float32',
        transform=rio.Affine(9000.0, 0.0, -12048530.45, 0.0, -9000.0, 5568540.83),
        crs='+proj=cea +lon_0=0 +lat_ts=30 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
    )

    out_dst.write(np.expand_dims(out.reshape(DIMS), axis=0))
    out_dst.close()


def save_all_preds(target_dir, in_features, mod_f, out_dir, remove, cuda, lead_time):

    info = parse_fname(mod_f)

    targets = sorted(glob.glob(os.path.join(target_dir, '*.dat')))
    if remove:
        targets = [x for x in targets if ('/2007' in x or '/2015' in x or '/2017' in x)]

    dat = AggregateAllSpatial(targets[50], in_features, lead_time=lead_time, n_weeks=int(info['nMonths']))

    dat = dat.premake_features()
    model = make_model(mod_f, cuda, dat.shape[0])
    
    dt = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    for f in targets:
        try:
            print(f)
            out = get_pred_true_arrays(model, mod_f, f, in_features, lead_time, dt)
            save_arrays(out_dir, out, f)
        except AssertionError as e:
            print(e, '\nSkipping this target')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model for entire domain for all target images.')

    parser.add_argument('-mf', '--model_file', type=str, help='Path to pickled model file.')
    parser.add_argument('-td', '--target_dir', type=str, help='Directory containing memmaps of all target images.')
    parser.add_argument('-if', '--in_features', type=str, help='Directory contining all memmap input features.')
    parser.add_argument('-od', '--out_dir', type=str, help='Directory to write np arrays out to.')
    parser.add_argument('-lt', '--lead_time', type=int, help='Lead time in weeks.')

    cuda = True if torch.cuda.is_available() else False
    args = parser.parse_args()

    save_all_preds(mod_f=args.model_file, target_dir=args.target_dir, in_features=args.in_features,
                   out_dir=args.out_dir, remove=True, cuda=cuda, lead_time=args.lead_time)
