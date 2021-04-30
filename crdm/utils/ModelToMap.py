import argparse
import os
import numpy as np
from crdm.models.SeqToSeq import Seq2Seq
from crdm.loaders.AggregatePixels import PremakeTrainingPixels
from crdm.utils.ImportantVars import DIMS, LENGTH, holdouts
import pickle
from pathlib import Path
import rasterio as rio
import torch
from tqdm import tqdm

BATCH = 2488


class Mapper(object):

    def __init__(self, model, metadata, features, classes, out_dir, shps, test=True, holdout=None, categorical=False):

        self.model = model
        self.features = features
        self.classes = classes
        self.out_dir = out_dir
        self.test = test
        self.metadata = metadata
        self.shps = shps
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.holdout = holdout
        self.categorical = categorical

        targets = sorted([str(x) for x in Path(classes).glob('*.dat')])
        targets = [targets[i:i + metadata['mx_lead']] for i in range(len(targets))]
        targets = list(filter(lambda x: len(x) == metadata['mx_lead'], targets))
        targets = [x for x in targets if ('/201707' in x[0])] if test else targets

        self.targets = targets
        self.indices = list(range(0, LENGTH+1, BATCH))

    def get_preds(self):

        fill_shp = (BATCH, self.shps['train_x.dat'][-1])

        for target in self.targets:
            print(target[0])
            x_out = []
            try:
                agg = PremakeTrainingPixels(in_features=self.features, targets=target,
                                            n_weeks=self.metadata['n_weeks'])

                x_full, _ = agg.premake_features(list(range(self.indices[0], self.indices[-1])))

            except AssertionError as e:
                print(e)
                continue
            for i in tqdm(range(len(mapper.indices)-1)):

                idx = list(range(mapper.indices[i], mapper.indices[i+1]))

                x = x_full[:, :, idx]

                # Batch, Seq, Feature
                x = x.swapaxes(0, 2)
                # Replace variable with a random value if specified in init.
                if mapper.holdout is not None:
                    x[:, :, holdouts[mapper.holdout]] = np.random.uniform(-1, 1, fill_shp)
                x = mapper.dtype(x)

                outputs = mapper.model(x)
                outputs = outputs.detach().cpu().numpy()
                outputs = np.argmax(outputs, 1) if self.categorical else outputs

                x_out.append(outputs)

            x = np.concatenate(x_out, axis=0)
            x = x.swapaxes(0, 1).reshape(self.metadata['mx_lead'], *DIMS)
            x = x if self.categorical else np.clip(np.round(x * 5), 0, 5)
            self.save_arrays(x, target)

    def save_arrays(self, data, target):

        suffix = 'preds_{}.tif'.format(self.holdout)
        dt = 'int8'
        out_dst = rio.open(
            os.path.join(self.out_dir, os.path.basename(target[0]).replace('USDM.dat', suffix)),
            'w',
            driver='GTiff',
            height=DIMS[0],
            width=DIMS[1],
            count=self.metadata['mx_lead'],
            dtype=dt,
            transform=rio.Affine(9000.0, 0.0, -12048530.45, 0.0, -9000.0, 5568540.83),
            crs='+proj=cea +lon_0=0 +lat_ts=30 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
        )

        out_dst.write(data.astype(np.int8))
        out_dst.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model for entire domain for all target images.')

    parser.add_argument('-md', '--model_dir', type=str, help='Path to the directory containing model results.')
    parser.add_argument('-t', '--targets', type=str, help='Directory containing memmaps of all target images.')
    parser.add_argument('-f', '--features', type=str, help='Directory contining all memmap input features.')
    parser.add_argument('-ho', '--holdout', type=str, default=None,
                        help='Which variable should be held out to run the model')

    args = parser.parse_args()

    shps = pickle.load(open(os.path.join(args.model_dir, 'shps.p'), 'rb'))
    metadata = pickle.load(open(os.path.join(args.model_dir, 'metadata_0_seq.p'), 'rb'))

    model = Seq2Seq(1, shps['train_x.dat'][1], shps['train_x.dat'][-1],
                    metadata['hidden_size'], metadata['mx_lead'], categorical=False)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model_0_seq.p'), map_location=torch.device(device)))

    if torch.cuda.is_available():
        print('GPU')
        model.cuda()

    out_dir = os.path.join(args.model_dir, 'preds')

    mapper = Mapper(model, metadata, args.features, args.targets, out_dir, shps, True, args.holdout, False)
    mapper.get_preds()
