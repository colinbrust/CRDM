import argparse
import torch
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from dc.classification.TrainLSTMCategorical import LSTM
from dc.loaders.AggregateAllPixels import AggregateAllPixles
from dc.utils.ImportantVars import DIMS, LENGTH, MONTHLY_VARS, WEEKLY_VARS
from dc.utils.ParseFileNames import parse_fname
import rasterio as rio


def make_model(mod_f, init, cuda, continuous):

    info = parse_fname(mod_f)
    const_size = 11
    out_size = 1 if continuous else 6

    weekly_size = len(WEEKLY_VARS) + 1 if init else len(WEEKLY_VARS)
    # make model from hyperparams and load trained parameters.
    model = LSTM(weekly_size=weekly_size, monthly_size=len(MONTHLY_VARS),
                 hidden_size=int(info['hiddenSize']), output_size=out_size,
                 batch_size=int(info['batch']), const_size=const_size, cuda=cuda, num_layers=int(info['numLayers']))

    model.load_state_dict(torch.load(mod_f)) if cuda and torch.cuda.is_available() else model.load_state_dict(
        torch.load(mod_f, map_location=torch.device('cpu')))
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model


def get_pred_true_arrays(model, mod_f, target, in_features, init, cuda, continuous):

    # Get hyperparameters from filename
    info = parse_fname(mod_f)
    stateful, batch, nWeeks = bool(info['stateful']), int(info['batch']), int(info['nWeeks'])
    data = AggregateAllPixles(targets=target, in_features=in_features, n_weeks=nWeeks, init=init)

    weeklys, monthlys, constants = data.premake_features()

    constants = constants.swapaxes(0, 1)
    monthlys = monthlys.swapaxes(2, 0)
    weeklys = weeklys.swapaxes(2, 0)

    batch_indices = [x for x in range(0, LENGTH, batch)]
    tail = [(LENGTH) - batch, LENGTH + 1]

    all_preds = []

    if stateful:
        week_h, week_c = model.init_state()
        month_h, month_c = model.init_state()

    for i in range(len(batch_indices) - 1):
        week_batch = weeklys[batch_indices[i]: batch_indices[i + 1]].swapaxes(0, 1)
        mon_batch = monthlys[batch_indices[i]: batch_indices[i + 1]].swapaxes(0, 1)
        const_batch = constants[batch_indices[i]: batch_indices[i + 1]]

        if not stateful:
            week_h, week_c = model.init_state()
            month_h, month_c = model.init_state()

        preds, (week_h, week_c), (month_h, month_c) = model(
            torch.tensor(week_batch).type(
                torch.cuda.FloatTensor if (torch.cuda.is_available() and cuda) else torch.FloatTensor),
            torch.tensor(mon_batch).type(
                torch.cuda.FloatTensor if (torch.cuda.is_available() and cuda) else torch.FloatTensor),
            torch.tensor(const_batch).type(
                torch.cuda.FloatTensor if (torch.cuda.is_available() and cuda) else torch.FloatTensor),
            (week_h, week_c), (month_h, month_c)
        )

        if stateful:
            week_h, month_h = week_h.detach(), month_h.detach()
            week_c, month_c = week_c.detach(), month_c.detach()

        if continuous:
            preds = [x.cpu().detach().numpy() if cuda else x.detach().numpy() for x in preds]
        else:
            preds = [np.argmax(x.cpu().detach().numpy(), axis=1) if cuda else np.argmax(x.detach().numpy(), axis=1) for x in preds]

        all_preds.append(preds)

    week_batch = weeklys[tail[0]: tail[1]].swapaxes(0, 1)
    mon_batch = monthlys[tail[0]: tail[1]].swapaxes(0, 1)
    const_batch = constants[tail[0]: tail[1]]

    week_h, week_c = model.init_state()
    month_h, month_c = model.init_state()

    preds, _, _ = model(
        torch.tensor(week_batch).type(
            torch.cuda.FloatTensor if (torch.cuda.is_available() and cuda) else torch.FloatTensor),
        torch.tensor(mon_batch).type(
            torch.cuda.FloatTensor if (torch.cuda.is_available() and cuda) else torch.FloatTensor),
        torch.tensor(const_batch).type(
            torch.cuda.FloatTensor if (torch.cuda.is_available() and cuda) else torch.FloatTensor),
        (week_h, week_c), (month_h, month_c)
    )

    if continuous:
        preds = [x.cpu().detach().numpy() if cuda else x.detach().numpy() for x in preds]
    else:
        preds = [np.argmax(x.cpu().detach().numpy(), axis=1) if cuda else np.argmax(x.detach().numpy(), axis=1) for x in preds]

    fill = LENGTH - (len(all_preds) * batch)
    fill = [x[-fill:] for x in preds]

    all_preds.append(fill)
    
    out = np.concatenate([*all_preds], axis=1)
    out = out.astype('float32') if continuous else out.astype('int8')

    return out


def save_arrays(out_dir, out, target, continuous):

    dt = 'float32' if continuous else 'int8'

    out_dst = rio.open(
        os.path.join(out_dir, os.path.basename(target[0]).replace('_USDM.dat', '_preds.tif')),
        'w',
        driver='GTiff',
        height=DIMS[0],
        width=DIMS[1],
        count=4,
        dtype=dt,
        transform=rio.Affine(9000.0, 0.0, -12048530.45, 0.0, -9000.0, 5568540.83),
        crs='+proj=cea +lon_0=0 +lat_ts=30 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
    )

    [out_dst.write(x.reshape(DIMS), i) for i, x in enumerate(out, 1)]
    out_dst.close()


def save_all_preds(target_dir, in_features, mod_f, out_dir, remove, init, cuda, continuous):

    model = make_model(mod_f, init, cuda, continuous=continuous)

    targets_tmp = sorted(glob.glob(os.path.join(target_dir, '*.dat')))
    targets = []
    for i in range(len(targets_tmp)):
        if remove:
            files = targets_tmp[i:i + 8]
            if any(['/2015' in x or '/2017' in x for x in files]):
                targets.append(files)
            else:
                continue
        else:
            targets.append(targets_tmp[i:i + 8])

    for f in targets:
        try:
            out = get_pred_true_arrays(model, mod_f, f, in_features, init, cuda, continuous)
            save_arrays(out_dir, out, f, continuous)
        except AssertionError as e:
            print(e, '\nSkipping this target')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run model for entire domain for all target images.')

    parser.add_argument('-mf', '--model_file', type=str, help='Path to pickled model file.')
    parser.add_argument('-td', '--target_dir', type=str, help='Directory containing memmaps of all target images.')
    parser.add_argument('-if', '--in_features', type=str, help='Directory contining all memmap input features.')
    parser.add_argument('-od', '--out_dir', type=str, help='Directory to write np arrays out to.')
    parser.add_argument('--rm', dest='remove', action='store_true', help='Run model for only 2015 and 2017 (test years not used for training).')
    parser.add_argument('--no-rm', dest='remove', action='store_false', help='Run model for all years.')
    parser.set_defaults(remove=False)

    parser.add_argument('--init', dest='init', action='store_true', help='Use initial drought condition as model input.')
    parser.add_argument('--no-init', dest='init', action='store_false', help='Do not use initial drought condition as model input..')
    parser.set_defaults(init=False)

    parser.add_argument('--cont', dest='cont', action='store_true', help='Use model that predicts continuous drought.')
    parser.add_argument('--no-cont', dest='cont', action='store_false', help='Use model that predicts categorical drought.')
    parser.set_defaults(cont=False)

    cuda = True if torch.cuda.is_available() else False
    args = parser.parse_args()

    save_all_preds(mod_f=args.model_file, target_dir=args.target_dir, in_features=args.in_features,
                   out_dir=args.out_dir, remove=args.remove, init=args.init, cuda=cuda, continuous=args.cont)
