import argparse
from dc.fetch.DownloadFile import download_file
import os
from pathlib import Path


def fetch_gridmet(download_dir, variable):

    print(variable)
    # Get date of the last available raster
    f_dir = [str(x) for x in Path(download_dir).glob(variable + '*.nc')]
    years = [int(os.path.basename(x)[-7:-3]) for x in f_dir] if f_dir else [2002]

    # Make a list of years for which we don't have data
    missing_years = [x for x in range(max(years), 2021)]

    os.chdir(download_dir)

    # Gridmet data location
    url = 'http://www.northwestknowledge.net/metdata/data/'

    # Gridmet data are stacked as one NetCDF per year. Make a list of urls with all years that we have missing data
    urls = [url + variable + '_' + str(x) + '.nc' for x in missing_years]

    for f in urls:
        if not os.path.exists(os.path.basename(f)):
            download_file(f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Download SMAP Data')
    parser.add_argument('-od', '--out_dir', type=str, default='.', help='directory to write data to.')

    args = parser.parse_args()

    for v in ['tmmx', 'tmmn', 'pr', 'vs', 'rmax', 'rmin', 'srad', 'vpd']:
        fetch_gridmet(args.od, v)
