import argparse
import torch
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from crdm.classification.TrainLSTM import LSTM
from crdm.loaders.AggregateAllPixels import AggregateAllPixles
from crdm.utils.ImportantVars import DIMS, LENGTH
from crdm.utils.ParseFileNames import parse_fname


def make_model(mod_f):

    info = parse_fname(mod_f)
    const_size = 16 if info['rmFeatures'] == 'True' else 18
    # make model from hyperparams and load trained parameters.
    
    model = LSTM(input_size=12, hidden_size=int(info['hiddenSize']), output_size=6, 
                 batch_size=int(info['batch']), seq_len=int(info['nMonths']), const_size=const_size)
    model.load_state_dict(torch.load(mod_f))
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model

def get_pred_true_arrays(model, mod_f, target, in_features):
    # TODO: Add const_size and input_size as additional filename descriptors.
    # TODO: Make helper to extract epochs, batch, etc from data.
    # Get hyperparameters from filename

    info = parse_fname(mod_f)
    batch, nMonths, leadTime = int(info['batch']), int(info['nMonths']), int(info['leadTime'])

    data = AggregateAllPixles(target=target, in_features=in_features, 
                              lead_time=leadTime, n_months=nMonths)
    
    if info['rmFeatures'] == 'True':
        data.remove_lat_lon()

    monthlys, constants = data.premake_features()
    constants = np.nan_to_num(constants, nan=-0.5)
    monthlys = np.nan_to_num(monthlys, nan=-0.5)

    constants = constants.swapaxes(0, 1)
    monthlys = monthlys.swapaxes(2, 0)

    batch_indices = [x for x in range(0, LENGTH, batch)]
    tail = [(LENGTH) - batch, LENGTH + 1]

    all_preds = np.array([])

    for i in range(len(batch_indices) - 1):

        mon_batch = monthlys[batch_indices[i]: batch_indices[i+1]].swapaxes(0, 1)
        const_batch = constants[batch_indices[i]: batch_indices[i+1]]
        preds = model(torch.Tensor(mon_batch), torch.Tensor(const_batch))
        preds = np.argmax(preds.detach().numpy(), axis=1)

        all_preds = np.concatenate((all_preds, preds))

    mon_batch = monthlys[tail[0]: tail[1]].swapaxes(0, 1)
    const_batch = constants[tail[0]: tail[1]]
    preds = model(torch.Tensor(mon_batch), torch.Tensor(const_batch))
    preds = np.argmax(preds.detach().numpy(), axis=1)
    fill = LENGTH - len(all_preds)
    fill = preds[-fill:]

    all_preds = np.concatenate((all_preds, fill))
    out = all_preds.reshape(DIMS)
    actual = np.memmap(target, dtype='int8', mode='c', shape=DIMS)

    return {'preds': out,
            'valid': actual}


def save_arrays(out_dir, out_dict, target, mod_f):

    base = os.path.basename(target).replace('.dat', '')
    mod_name = os.path.basename(mod_f).replace('.p', '')
    np.savetxt(os.path.join(out_dir, '{}_{}_pred.csv'.format(base, mod_name)), out_dict['preds'], delimiter=',')
    np.savetxt(os.path.join(out_dir, '{}_{}_real.csv'.format(base, mod_name)), out_dict['valid'], delimiter=',')


def save_all_preds(target_dir, in_features, mod_f, out_dir, test):

    model = make_model(mod_f)
    f_list = [str(x) for x in Path(target_dir).glob('*_USDM.dat')]
    f_list = [x for x in f_list if ('/2015' in x or '/2016' in x)] if test else f_list

    for f in f_list:
        print(f)
        try:
            out_dict = get_pred_true_arrays(model, mod_f, f, in_features)
            save_arrays(out_dir, out_dict, f, mod_f)
        except AssertionError as e:
            print(e, '\nSkipping this target')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model for entire domain for all target images.')

    parser.add_argument('-mf', '--model_file', type=str, help='Path to pickled model file.')
    parser.add_argument('-td', '--target_dir', type=str, help='Directory containing memmaps of all target images.')
    parser.add_argument('-if', '--in_features', type=str, help='Directory contining all memmap input features.')
    parser.add_argument('-od', '--out_dir', type=str, help='Directory to write np arrays out to.')
    parser.add_argument('--test', dest='test', action='store_true', help='Run model for only 2015 and 2016 (test years not used for training).')
    parser.add_argument('--no-test', dest='test', action='store_false', help='Run model for all years.')
    parser.set_defaults(test=False)

    args = parser.parse_args()

    save_all_preds(mod_f=args.model_file, target_dir=args.target_dir, 
                   in_features=args.in_features, out_dir=args.out_dir, test = args.test)
