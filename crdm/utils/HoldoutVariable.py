import argparse
import torch
import os
import glob
import numpy as np
from crdm.training.TrainLSTM import LSTM
from crdm.loaders.EvalLSTM import AggregateAllPixles
from crdm.utils.ImportantVars import DIMS, LENGTH, MONTHLY_VARS, WEEKLY_VARS
from crdm.utils.ParseFileNames import parse_fname
import rasterio as rio

CONST_SIZE = 20
OUT_SIZE = 1

def make_model(mod_f, cuda, bs):
    info = parse_fname(mod_f)

    weekly_size = len(WEEKLY_VARS) + 4

    # make model from hyperparams and load trained parameters.
    model = LSTM(weekly_size=weekly_size, monthly_size=len(MONTHLY_VARS),
                 hidden_size=int(info['hiddenSize']), output_size=OUT_SIZE,
                 batch_size=bs, const_size=CONST_SIZE, cuda=cuda)

    model.load_state_dict(torch.load(mod_f)) if cuda and torch.cuda.is_available() else model.load_state_dict(
        torch.load(mod_f, map_location=torch.device('cpu')))

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model


def get_pred_true_arrays(model, mod_f, target, in_features, init, cuda, continuous, var_type, idx, lead_time):

    # Get hyperparameters from filename
    info = parse_fname(mod_f)
    batch, nWeeks = int(info['batch']), int(info['nMonths'])
    data = AggregateAllPixles(target=target, lead_time=lead_time,  in_features=in_features, n_weeks=nWeeks, init=init)

    weeklys, monthlys, constants = data.premake_features()

    constants = constants.swapaxes(0, 1)
    monthlys = monthlys.swapaxes(2, 0)
    weeklys = weeklys.swapaxes(2, 0)

    batch_indices = [x for x in range(0, LENGTH, batch)]
    tail = [(LENGTH) - batch, LENGTH + 1]

    all_preds = []

    for i in range(len(batch_indices) - 1):
        week_batch = weeklys[batch_indices[i]: batch_indices[i + 1]].swapaxes(0, 1)
        mon_batch = monthlys[batch_indices[i]: batch_indices[i + 1]].swapaxes(0, 1)
        const_batch = constants[batch_indices[i]: batch_indices[i + 1]]

        if var_type == 'weekly':
            new = week_batch[..., idx]
            new[new != -1.5] = np.random.uniform(-1, 1, len(new[new != -1.5]))
            week_batch[..., idx] = new
            week_batch = torch.cuda.FloatTensor(week_batch) if cuda else torch.FloatTensor(week_batch)
        elif var_type == 'monthly':
            new = mon_batch[..., idx]
            new[new != -1.5] = np.random.uniform(-1, 1, len(new[new != -1.5]))
            mon_batch[..., idx] = new
            mon_batch = torch.cuda.FloatTensor(mon_batch) if cuda else torch.FloatTensor(mon_batch)
        else:
            pass

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

        preds = preds.cpu().detach().numpy()

        all_preds.append(preds)

    week_batch = weeklys[tail[0]: tail[1]].swapaxes(0, 1)
    mon_batch = monthlys[tail[0]: tail[1]].swapaxes(0, 1)
    const_batch = constants[tail[0]: tail[1]]

    if var_type == 'weekly':
        new = week_batch[..., idx]
        new[new != -1.5] = np.random.uniform(-1, 1, len(new[new != -1.5]))
        week_batch[..., idx] = new
        week_batch = torch.cuda.FloatTensor(week_batch) if cuda else torch.FloatTensor(week_batch)
    elif var_type == 'monthly':
        new = mon_batch[..., idx]
        new[new != -1.5] = np.random.uniform(-1, 1, len(new[new != -1.5]))
        mon_batch[..., idx] = new
        mon_batch = torch.cuda.FloatTensor(mon_batch) if cuda else torch.FloatTensor(mon_batch)
    else:
        pass
    
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

    preds = preds.cpu().detach().numpy()

    fill = LENGTH - (len(all_preds) * batch)
    fill = preds[-fill:]

    all_preds.append(fill)

    out = np.concatenate([*all_preds])
    out = out.astype('float32') if continuous else out.astype('int8')

    return out


def save_arrays(out_dir, out, target, continuous, var, lead_time):
    dt = 'float32' if continuous else 'int8'

    out_dst = rio.open(
        os.path.join(out_dir, os.path.basename(target).replace('_USDM.dat', '_') + var + '_' + str(lead_time) + '.tif'),
        'w',
        driver='GTiff',
        height=DIMS[0],
        width=DIMS[1],
        count=1,
        dtype=dt,
        transform=rio.Affine(9000.0, 0.0, -12048530.45, 0.0, -9000.0, 5568540.83),
        crs='+proj=cea +lon_0=0 +lat_ts=30 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
    )

    out_dst.write(np.expand_dims(out.reshape(DIMS), axis=0))
    out_dst.close()


def save_all_preds(target_dir, in_features, mod_f, out_dir, remove, init, cuda, continuous, full=False):
    model = make_model(mod_f, cuda, 1024)

    targets_tmp = sorted(glob.glob(os.path.join(target_dir, '*.dat')))
    targets = [x for x in targets_tmp if '/2015' in x or '/2017' in x or '/2007' in x]
    
    weeklys = WEEKLY_VARS + ['USDM']
    

    for f in targets:
        for lead_time in [2, 4, 6, 8]:

            print('{} - Holding out {} - Lead Time {}'.format(f, 'full', lead_time))
            out = get_pred_true_arrays(model, mod_f, f, in_features, True, True, True, 'full', None, lead_time)
            save_arrays(out_dir, out, f, True, 'full', lead_time)
            

            if not full:
                for idx, var in enumerate(weeklys):
                    print('{} - Holding out {} - Lead Time {}'.format(f, var, lead_time))
                    out = get_pred_true_arrays(model, mod_f, f, in_features, True, True, True, 'weekly', idx, lead_time)
                    save_arrays(out_dir, out, f, True, var, lead_time)

                for idx, var in enumerate(MONTHLY_VARS):
                    print('{} - Holding out {} - Lead Time {}'.format(f, var, lead_time))
                    out = get_pred_true_arrays(model, mod_f, f, in_features, True, True, True, 'monthly', idx, lead_time)
                    save_arrays(out_dir, out, f, True, var, lead_time)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run model for entire domain for all target images.')

    parser.add_argument('-mf', '--model_file', type=str, help='Path to pickled model file.')
    parser.add_argument('-td', '--target_dir', type=str, help='Directory containing memmaps of all target images.')
    parser.add_argument('-if', '--in_features', type=str, help='Directory contining all memmap input features.')
    parser.add_argument('-od', '--out_dir', type=str, help='Directory to write np arrays out to.')
    parser.add_argument('--full', dest='full', action='store_true', help='Only run full model. Dont holdout variables.')
    parser.add_argument('--no-full', dest='full', action='store_false', help='Holdout all model variables.')

    parser.set_defaults(full=False)
    cuda = True if torch.cuda.is_available() else False
    args = parser.parse_args()

    save_all_preds(mod_f=args.model_file, target_dir=args.target_dir, in_features=args.in_features,
                   out_dir=args.out_dir, remove=True, init=True, cuda=cuda, continuous=True, full=args.full)
