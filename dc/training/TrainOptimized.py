import argparse
from dc.training.TrainSeq import train_seq
from dc.utils.ModelToMap import Mapper
from dc.utils.ImportantVars import ConvergenceError
import os
import pickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Drought Prediction Model')
    parser.add_argument('-if', '--in_features', type=str, help='Directory containing training features.')
    parser.add_argument('-c', '--out_classes', type=str, help='Directory containing target classes.')
    parser.add_argument('-e', '--epochs', type=int, default=25, help='Number of epochs.')
    parser.add_argument('-dn', '--dirname', type=str, default=None,
                        help='Directory with training data to use. If left blank, training data will be created.')

    args = parser.parse_args()

    # These are the best performing hyperparameters.
    setup = {
        'in_features': args.in_features,
        'out_classes': args.out_classes,
        'epochs': args.epochs,
        'batch_size': 128,
        'hidden_size': 128,
        'n_weeks': 30,
        'mx_lead': 12,
        'size': 1024,
        'lead_time': None,
        'early_stop': 20,
        'categorical': False,
        'pix_mask': '/mnt/e/PycharmProjects/DroughtCast/data/pix_mask.dat',
        'model_type': 'vanilla',
        'dirname': args.dirname,
        'iter_print': 100
    }
    
    with open(os.path.join(setup['dirname'], 'shps.p'), 'rb') as f:
        shps = pickle.load(f)
    
    for ensemble in range(5):
    
        i = 0

        while os.path.exists('ensemble_{}'.format(i)):
            i += 1

        out_dir = 'ensemble_{}'.format(i)
        os.mkdir(out_dir)
        setup['out_dir'] = out_dir

        flag = True
        while flag:
            try:
                model = train_seq(setup)
                flag = False
            except ConvergenceError as e:
                print(e)

        mpr = Mapper(model, setup, args.in_features, args.out_classes, out_dir, shps, True, None, setup['categorical'])
        mpr.get_preds()