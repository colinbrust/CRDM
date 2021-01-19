from crdm.classification.TrainLSTM import train_lstm
from crdm.utils.MakeLSTMPixelTS import make_lstm_pixel_ts
import os


def run_different_months(target_dir, in_features, premade_dir):
    for n_months in range(5, 14):
        base_name = make_lstm_pixel_ts(target_dir, in_features, 1, 15000, n_months=n_months, 
                                    out_dir=premade_dir, rm_features=True)

        mon_f = os.path.join(premade_dir, 'featType-monthly'+base_name)
        const_f = os.path.join(premade_dir, 'featType-constant'+base_name)
        target_f = os.path.join(premade_dir, 'featType-target'+base_name)

        train_lstm(const_f, mon_f, target_f, epochs=25, batch_size=128, hidden_size=512)


run_different_months('../out_classes/out_memmap', '../in_features', '../premade')