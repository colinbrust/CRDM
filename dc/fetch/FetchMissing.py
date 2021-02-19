import subprocess
import os
import requests
import datetime as dt
from pathlib import Path
from dateutil.relativedelta import relativedelta
from typing import List
from abc import ABC, abstractmethod


class FetchMissing(ABC):

    def __init__(self, download_dir, variable, **kwargs):
        self.base_dir = os.getcwd()
        self.download_dir = download_dir
        self.variable = variable
        self.missing = self.get_missing()
        self.tmp_files = []
        self.kwargs = kwargs

    @staticmethod
    def download_file(url):
        print(url)
        f_name = os.path.basename(url)
        r = requests.get(url, stream=True)

        if r.status_code == requests.codes.ok:
            with open(f_name, 'wb') as f:
                for data in r:
                    f.write(data)

    def get_missing(self) -> List[dt.date]:

        f_dir = [str(x) for x in Path(self.download_dir).glob('*' + self.variable + '.tif')]
        dates = [os.path.basename(x)[:8] for x in f_dir] if f_dir else ['20011231']
        dates = [dt.datetime.strptime(x, '%Y%m%d') for x in dates]
        latest = max(dates).date()
        today = dt.date.today().replace(day=1)
        diff = today-latest

        return [today - relativedelta(days=x) for x in range(1, diff.days)]

    @abstractmethod
    def fetch_data(self):
        pass

    @abstractmethod
    def standardize_format(self, template):
        pass

    def cleanup(self):
        [os.remove(x) for x in self.tmp_files]

    def fetch_standardize_save(self, template):
        if self.missing:
            self.fetch_data()
            self.standardize_format(template=template)
            self.cleanup()
            os.chdir(self.base_dir)
        else:
            print('{} data are up to date.'.format(self.variable))
