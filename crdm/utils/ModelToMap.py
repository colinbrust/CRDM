import torch
import os
import matplotlib.pyplot as plt
from crdm.classification.TrainLSTM import LSTM
from crdm.loaders.AggregateAllPixels import AggregateAllPixles
from crdm.utils.ImportantVars import DIMS, LENGTH


def make_model(mod_f):

    epochs, batch, nMonths, hiddenSize, leadTime = [int(x.split('-')[-1]) for x in os.path.basename(f).split('_')[1:-1]]
    # make model from hyperparams and load trained parameters.
    model = LSTM(input_size=12, hidden_size=hiddenSize, output_size=6, 
                batch_size=batch, seq_len=nMonths, const_size=18)
    model.load_state_dict(torch.load(mod_f))
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model

def get_pred_true_arrays(model, mod_f, target, in_features):
    # TODO: Add const_size and input_size as additional filename descriptors.
    # TODO: Make helper to extract epochs, batch, etc from data.
    # Get hyperparameters from filename
    epochs, batch, nMonths, hiddenSize, leadTime = [int(x.split('-')[-1]) for x in os.path.basename(f).split('_')[1:-1]]

    data = AggregateAllPixles(target=target, in_features=in_features, 
                            lead_time=leadTime, n_months=nMonths)

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


def save_arrays(out_dir, out_dict, target):

    base = os.path.basename(target).replace('.dat', '')
    np.savetxt(os.path.join(out_dir, '{}_pred.csv'.format(base)), out_dict['preds'], delimiter=',')
    np.savetxt(os.path.join(out_dir, '{}_real.csv'.format(base)), out_dict['valid'], delimiter=',')


mod_f = '/Users/colinbrust/projects/CRDM/data/drought/model_results/LSTM_epochs-20_batch-32_nMonths-13_hiddenSize-32_leadTime-2_model.p'
target = '/Users/colinbrust/projects/CRDM/data/drought/out_classes/out_memmap/20130702_USDM.dat'
in_features = '/Users/colinbrust/projects/CRDM/data/drought/in_features'

