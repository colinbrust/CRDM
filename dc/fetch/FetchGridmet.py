import datetime as dt
from dc.fetch.DownloadFile import download_file
from dc.utils.ImportantVars import usdm_dates
import numpy as np
import os
from pathlib import Path
import rasterio as rio
from dateutil import relativedelta as rd
from shutil import rmtree


def make_date_range(start_date, end_date):
    num_days = (end_date - start_date).days
    return [start_date + rd.relativedelta(days=x) for x in range(1, num_days + 1)]


mapping_dict = {
    'tmmx': ' daily_maximum_temperature',
    'tmmn': 'daily_minimum_temperature',
    'pr': 'precipitation_amount',
    'vs': 'daily_mean_wind_speed',
    'rmax': 'daily_maximum_relative_humidity',
    'rmin': 'daily_mainimum_relative_humidity',
    'srad': 'daily_mean_shortwave_radiation_at_surface',
    'vpd': 'daily_mean_vapor_pressure_deficit'
}


def fetch_gridmet(download_dir, variable, nodata: int = 32767, save: bool = False):
    # Make temporary directory for holding daily images when making weekly means.
    if os.path.exists('./temp'):
        print('"./temp" already exists. Please remove or move to a new directory.')
        return None
    else:
        os.makedirs('./temp')

    # Get date of the last available raster
    f_list = [str(x) for x in Path(download_dir).glob(f'*{variable}.tif')]

    # Get latest raster that we have.
    if f_list:
        start_date = max([dt.datetime.strptime(os.path.basename(x)[:8], '%Y%m%d').date() for x in f_list])
    else:
        start_date = dt.date(2002, 6, 1)

    to_download = [x for x in usdm_dates if x > start_date]
    for end_date in to_download:
        days = make_date_range(start_date, end_date)

        for day in days:
            url = f'http://thredds.northwestknowledge.net:8080/thredds/ncss/agg_met_{variable}_1979_CurrentYear_CONUS' \
                  f'.nc?var={mapping_dict[variable]}&horizStride=1' \
                  f'&time={day.strftime("%Y-%m-%d")}T00%3A00%3A00Z&accept=netcdf'

            download_file(url, Path('temp') / f'{variable}_{str(day)}.tif')

        arr = np.array([rio.open(x).read(1) for x in Path('temp').glob('*.tif')])
        arr = np.mean(arr, axis=0)
        arr = np.where(arr == nodata, np.nan, arr)

        out_name = f"{str(day).replace('-', '')}_{variable}.tif"

        # Write out rio raster
    if not save:
        rmtree('./temp')




if __name__ == '__main__':
    for v in ['tmmx', 'tmmn', 'pr', 'vs', 'rmax', 'rmin', 'srad', 'vpd']:
        fetch_gridmet('/Users/colinbrust/projects/DroughtCast/data/in_features/tif/weekly', v)
