from crdm.utils.ImportantVars import LENGTH
from datetime import datetime as dt
from datetime import timedelta
import numpy as np
import os
import pickle
import torch
from torch.utils.data import Dataset


class DataStack(Dataset):

    def __init__(self, data_dir, n_weeks=20, train=True, lead_time=9999, cuda=True, num_samples=10000):

        assert train in ['train', 'test', 'all'], "Argument 'train' must be one of 'train', 'test', or 'all'."
        self.data_dir = data_dir
        self.n_weeks = n_weeks
        self.n_months = n_weeks // 4
        self.train = train
        self.lead_time = lead_time
        self.cuda = cuda
        self.num_samples = num_samples

        self.weekly, self.weekly_info = self.read('weekly')
        self.monthly, self.monthly_info = self.read('monthly')
        self.annual, self.annual_info = self.read('annual')
        self.constant, self.constant_info = self.read('constant')
        self.target, self.target_info = self.read('target')

        self.target_dates = self.target_info['dates']
        self.monthly_dates = self.monthly_info['dates']
        self.annual_dates = self.annual_info['dates']
        self.weekly_dates = self.get_valid_dates()

        if self.train == 'train':
            self.target_dates = [x for x in self.target_dates if not x.startswith(('2008', '2015', '2017'))]
        elif self.train == 'test':
            self.target_dates = [x for x in self.target_dates if x.startswith(('2008', '2015', '2017'))]
        else:
            pass

        self.indices = self.build_indices()

    def read(self, variable):
        dat = os.path.join(self.data_dir, variable+'.dat')
        info = os.path.join(self.data_dir, variable+'_info.dat')

        with open(info, 'rb') as pick:
            info = pickle.load(pick)

        dat = np.memmap(dat, dtype='float32', shape=info['shp'])
        dat = dat.squeeze()

        return dat, info

    def get_valid_dates(self):
        weekly_dates = self.weekly_info['dates']
        monthly_dates = self.monthly_info['dates']

        valid_dates = self.add_date_string(self.target_dates, False, False)

        out_dates = []

        for date in valid_dates:
            weekly_candidates = [date - timedelta(weeks=x) for x in range(1, self.n_weeks+8)]
            weekly_candidates = [str(x).replace('-', '') for x in weekly_candidates]
            monthly_candidates = list(set([x[:6]+'01' for x in weekly_candidates]))
            if all([x in weekly_dates for x in weekly_candidates]) and all([x in monthly_dates for x in monthly_candidates]):
                [out_dates.append(x) for x in weekly_candidates]

        weekly_dates = list(set(out_dates))
        weekly_dates = [str(x).replace('-', '') for x in weekly_dates]

        return sorted(weekly_dates)

    def build_indices(self):

        out_indices = []
        possible_dates = self.add_date_string(self.target_dates, False, True)

        for i in range(len(possible_dates)*4):

            target_date = self.target_dates[i // 4]
            target_idx = self.target_dates.index(target_date)

            lead_time = ((i % 4) + 1) * 2

            if self.lead_time != 9999 and lead_time != self.lead_time:
                continue

            target_date = self.add_date_string([target_date], False, False)[0]
            weekly_candidates = [(target_date - timedelta(weeks=lead_time)) - timedelta(weeks=x) for x in range(self.n_weeks)]
            weekly_candidates = [str(x).replace('-', '') for x in weekly_candidates]

            try:
                weekly_idx = [self.weekly_dates.index(x) for x in weekly_candidates]

                monthly_candidates = sorted(list(set([x[:6] + '01' for x in weekly_candidates])))
                monthly_candidates = monthly_candidates[-self.n_months:]
                monthly_idx = [self.monthly_dates.index(x) for x in monthly_candidates]

                annual_candidates = list(set([x[:4] + '0101' for x in weekly_candidates]))[0]
                annual_idx = self.annual_dates.index(annual_candidates)

                out_indices.append({
                    'weekly': weekly_idx,
                    'monthly': monthly_idx,
                    'annual': annual_idx,
                    'target': target_idx,
                    'lead_time': lead_time,
                    'target_date': target_date
                })
            except ValueError as e:
                print('{}\n Insufficient data for {}. Skipping training data for this date.'.format(e, target_date))

        return dict(zip(range(len(out_indices)), out_indices))

    @staticmethod
    def add_date_string(dates, add, as_str):
        dates = [dt.strptime(x, '%Y%m%d').date() for x in dates]
        dates = [x + timedelta(days=1) for x in dates] if add else [x - timedelta(days=1) for x in dates]

        if as_str:
            dates = [str(x).replace('-', '') for x in dates]

        return dates

    def holdout_variable(self, variable):
        # TODO: Swaps a variable with values selected from a uniform distribution ranging from -1 to 1
        pass

    def __getitem__(self, idx):

        batch = len(idx)
        idx = np.random.randint(0, len(self.indices), 1)
        indices = self.indices[idx[0]]
        samples = np.random.randint(0, LENGTH, batch)

        weekly = np.take(self.weekly, indices=indices['weekly'], axis=0)
        weekly = np.take(weekly, indices=samples, axis=-1)


        monthly = np.take(self.monthly, indices=indices['monthly'], axis=0)
        monthly = np.take(monthly, indices=samples, axis=-1)

        target = np.take(self.target, indices=indices['target'], axis=0)
        target = np.take(target, indices=samples, axis=-1).squeeze()

        # Append annual data to constants
        annual = np.take(self.annual, indices=indices['annual'], axis=0)
        annual = np.take(annual, indices=samples, axis=-1)
        annual = np.expand_dims(annual, axis=0)
        const = np.take(self.constant, indices=samples, axis=-1)

        const = np.concatenate((const, annual))
        lead_time = indices['lead_time']/8
        lead_time = np.ones_like(const[0]) * lead_time
        const = np.vstack((const, lead_time))

        dtype = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        return {
            'const': dtype(const).squeeze(),
            'mon': dtype(monthly).squeeze(),
            'week': dtype(weekly).squeeze(),
            'target': dtype(target).squeeze()
        }

    def __len__(self):
        return self.num_samples
