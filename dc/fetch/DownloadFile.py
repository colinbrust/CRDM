import os
import requests


def download_file(url, f_name):

    print(f'Downloading {f_name}...')
    r = requests.get(url, stream=True)

    if r.status_code == requests.codes.ok:
        with open(f_name, 'wb') as f:
            for data in r:
                f.write(data)
