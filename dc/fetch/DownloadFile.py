import os
import requests


def download_file(url):
    print(url)
    f_name = os.path.basename(url)
    r = requests.get(url, stream=True)

    if r.status_code == requests.codes.ok:
        with open(f_name, 'wb') as f:
            for data in r:
                f.write(data)
