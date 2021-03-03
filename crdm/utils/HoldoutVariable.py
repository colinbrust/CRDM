import argparse
from crdm.loaders.AggregateAllPixels import AggregateAllPixles
from crdm.utils.ImportantVars import LENGTH, MONTHLY_VARS, WEEKLY_VARS
from crdm.utils.ParseFileNames import parse_fname
from crdm.utils.ModelToMap import make_model
import torch
from torch import nn
import os
import glob
import numpy as np
import pickle


def get_error(model, mod_f, target, in_features, cuda, var_type, idx):

    assert var_type in ['monthly', 'weekly', 'full'], "Argument var_type must be one of 'full', 'monthly' or 'weekly'."

    # Get hyperparameters from filename
    info = parse_fname(mod_f)
    batch, nWeeks = int(info['batch']), int(info['nMonths'])

    data = AggregateAllPixles(target=target, lead_time=2,  in_features=in_features, n_weeks=nWeeks, init=True)

    weeklys, monthlys, constants = data.premake_features()

    constants = constants.swapaxes(0, 1)
    monthlys = monthlys.swapaxes(2, 0)
    weeklys = weeklys.swapaxes(2, 0)
    target = np.memmap(target, dtype='int8')/5

    batch_indices = [x for x in range(0, LENGTH, batch)]

    criterion = nn.MSELoss()
    err_out = []

    for i in range(len(batch_indices) - 1):
        week_batch = weeklys[batch_indices[i]: batch_indices[i + 1]].swapaxes(0, 1)
        mon_batch = monthlys[batch_indices[i]: batch_indices[i + 1]].swapaxes(0, 1)
        const_batch = constants[batch_indices[i]: batch_indices[i + 1]]
        target_batch = target[batch_indices[i]: batch_indices[i + 1]]
        target_batch = torch.cuda.FloatTensor(target_batch) if cuda else torch.FloatTensor(target_batch)

        # Replace non-NA values with a random value ranging from -1 to 1.
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

        loss = criterion(preds.squeeze(), target_batch.squeeze())
        err_out.append(loss.item())

    return err_out


def get_all_error(target_dir, in_features, mod_f, out_dir, cuda, bs):
    model = make_model(mod_f, cuda, bs)

    targets_tmp = sorted(glob.glob(os.path.join(target_dir, '*.dat')))
    targets = [x for x in targets_tmp if '/2015' in x or '/2017' in x]
    weeklys = WEEKLY_VARS + ['USDM']

    out_name = os.path.join(out_dir, 'holdout_dat.p')
    out_dict = {}
    for f in targets:

        f_dict = {}

        try:
            print(f)
            out = get_error(model, mod_f, f, in_features, cuda, 'full', None)
            f_dict['full'] = out

        except AssertionError as e:
            print(e, '\nSkipping this target')

        for idx, var in enumerate(weeklys):
            try:
                print('{} - Holding out {}'.format(f, var))
                out = get_error(model, mod_f, f, in_features, cuda, 'weekly', idx)
                f_dict[var] = out

            except AssertionError as e:
                print(e, '\nSkipping this target')

        for idx, var in enumerate(MONTHLY_VARS):
            try:
                print('{} - Holding out {}'.format(f, var))
                out = get_error(model, mod_f, f, in_features, cuda, 'monthly', idx)
                f_dict[var] = out

            except AssertionError as e:
                print(e, '\nSkipping this target')

        out_dict[f] = f_dict

        with open(out_name, 'wb') as pth:
            pickle.dump(out_dict, pth)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model for entire domain for all target images.')

    parser.add_argument('-mf', '--model_file', type=str, help='Path to pickled model file.')
    parser.add_argument('-td', '--target_dir', type=str, help='Directory containing memmaps of all target images.')
    parser.add_argument('-if', '--in_features', type=str, help='Directory contining all memmap input features.')
    parser.add_argument('-od', '--out_dir', type=str, help='Directory to write np arrays out to.')
    parser.add_argument('-bs', '--batch_size', type=int, help='Batch size to run evaluation at.')

    cuda = True if torch.cuda.is_available() else False
    args = parser.parse_args()

    get_all_error(target_dir=args.target_dir, in_features=args.in_features, mod_f=args.model_file,
                  out_dir=args.out_dir, cuda=cuda, bs=args.batch_size)
