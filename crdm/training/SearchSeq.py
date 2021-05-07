import argparse
from crdm.models.SeqAttn import Seq2Seq
from crdm.models.SeqVanilla import Seq2Seq as vanilla
from crdm.training.TrainModel import train_model
from crdm.loaders.LSTMLoader import LSTMLoader
from crdm.utils.ModelToMap import Mapper
import numpy as np
import os
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import pickle
from sklearn.utils.class_weight import compute_class_weight


def train_lstm(setup):

    with open(os.path.join(setup['dirname'], 'shps.p'), 'rb') as f:
        shps = pickle.load(f)

    # Make train and test set data loaders and add them to model setup
    train_loader = LSTMLoader(dirname=setup['dirname'], train=True, categorical=setup['categorical'],
                              n_weeks=setup['n_weeks'], sample=2000, mx_lead=setup['mx_lead'])# sample=131072)
    test_loader = LSTMLoader(dirname=setup['dirname'], train=False, categorical=setup['categorical'],
                             n_weeks=setup['n_weeks'], sample=2000, mx_lead=setup['mx_lead'])# sample=131072)
    setup['train'] = DataLoader(dataset=train_loader, batch_size=setup['batch_size'], shuffle=True, drop_last=True)
    setup['test'] = DataLoader(dataset=test_loader, batch_size=setup['batch_size'], shuffle=True, drop_last=True)

    if setup['model'] == 'vanilla':
        print('Using vanilla model.')
        setup['batch_first'] = True
        model = vanilla(1, shps['train_x.dat'][1], setup['hidden_size'], setup['mx_lead'], setup['categorical'])
    elif setup['model'] == 'attention':
        print('Using simple attention.')
        setup['batch_first'] = True
        model = Seq2Seq(1, shps['train_x.dat'][1], shps['train_x.dat'][-1],
                        setup['hidden_size'], setup['mx_lead'], categorical=setup['categorical'])
    else:
        raise ValueError("setup['model'] must be one of 'vanilla' or 'attention'.")

    if setup['categorical']:
        y = np.memmap(os.path.join(setup['dirname'], 'train_y.dat'), dtype='float32')
        y = y*5
        y = y.astype(np.int8)
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        print('Class Weights: {}'.format(class_weights))
        weights = torch.Tensor(class_weights).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

    criterion = nn.CrossEntropyLoss(weight=weights) if setup['categorical'] else nn.MSELoss()
    lr = 0.002
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5, verbose=True)

    setup['model'] = model
    setup['criterion'] = criterion
    setup['optimizer'] = optimizer
    setup['scheduler'] = scheduler

    model = train_model(setup)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Drought Prediction Model')
    parser.add_argument('-if', '--in_features', type=str, help='Directory containing training features.')
    parser.add_argument('-t', '--out_classes', type=str, help='Directory with targets')
    parser.add_argument('-e', '--epochs', type=int, default=25, help='Number of epochs.')
    parser.add_argument('-dn', '--dirname', type=str, default=None,
                        help='Directory with training data to use. If left blank, training data will be created.')

    parser.add_argument('--cat', dest='categorical', action='store_true', help='Treat targets as categorical')
    parser.add_argument('--no-cat', dest='categorical', action='store_false',
                        help="Treat targets as continuous")

    parser.add_argument('--van', dest='vanilla', action='store_true', help='Use Vanilla Seq2Seq model')
    parser.add_argument('--no-van', dest='vanilla', action='store_false', help="Use Seq2Seq with attetion")

    parser.set_defaults(search=False)

    args = parser.parse_args()

    setup = {'in_features': args.in_features, 'out_classes': args.out_classes, 'epochs': args.epochs,
             'early_stop': 10, 'categorical': args.categorical,
             'pix_mask': '/mnt/e/PycharmProjects/DroughtCast/data/pix_mask.dat', 
             'model': 'vanilla' if args.vanilla else 'attention', 'dirname': args.dirname}
    # global, con
    # global, cat
    # pix, cat
    # pix, con
    # pix, attn
    # pix, van

    with open(os.path.join(setup['dirname'], 'shps.p'), 'rb') as f:
        shps = pickle.load(f)

    for batch in [64, 128, 256, 512]:
        for hidden in [64, 128, 256, 512]:
            for history in [15, 30]:
                for lead in [8, 12]:
                    i = 0
                    while os.path.exists(os.path.join(setup['dirname'], 'metadata_{}.p'.format(i))):
                        i += 1

                    setup['index'] = i
                    setup['batch_size'] = batch
                    setup['hidden_size'] = hidden
                    setup['mx_lead'] = lead
                    setup['n_weeks'] = history

                    with open(os.path.join(setup['dirname'], 'metadata_{}.p'.format(setup['index'])), 'wb') as f:
                        pickle.dump(setup, f)

                    print(setup)
                    model = train_lstm(setup)
                    out_dir = os.path.join(setup['dirname'], 'preds_{}'.format(setup['index']))
                    mpr = Mapper(model, setup, args.in_features, args.out_classes, out_dir, shps, True, None, setup['categorical'])
                    mpr.get_preds()

