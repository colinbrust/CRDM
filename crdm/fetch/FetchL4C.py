from msc.utils.MatchProjection import reproject_image_to_template
import os
import subprocess as sp
from dateutil.relativedelta import relativedelta
import datetime as dt
from pathlib import Path
import rasterio as rio
import glob
import h5py
import calendar
import numpy as np
# example_url https://n5eil01u.ecs.nsidc.org/SMAP/SPL4CMDL.004/2015.04.01/SMAP_L4_C_mdl_20150401T000000_Vv4040_001.h5

# Make sure to run the following in the terminal so you can properly download data
# echo "machine urs.earthdata.nasa.gov login <uid> password <password>" >> ~/.netrc
# chmod 0600 ~/.netrc

# from FetchMissing import FetchMissing
class FetchL4C(FetchMissing):

    def fetch_data(self):
        os.chdir(self.tmp_dir)

        # Get list of all missing days
        max_date = (self.missing[0] + relativedelta(months=1)) - dt.timedelta(days=1)
        min_date = self.missing[-1]
        days = [str(max_date - dt.timedelta(days=x)).replace('-', '.') for x in range((max_date - min_date).days + 1)]

        for day in days:

            # Make valid URL to download from
            day_stripped = day.replace('.', '')
            url = 'https://n5eil01u.ecs.nsidc.org/SMAP/SPL4CMDL.004/{}/SMAP_L4_C_mdl_{}T000000_Vv4040_001.h5'.format(day, day_stripped)

            f_name = os.path.join(self.tmp_dir, os.path.basename(url))
            self.tmp_files.append(f_name)
            # If the file isn't already downloaded, download the file
            if not os.path.exists(f_name):
                self.download_file(url)

    def get_monthly_names(self, idx):
        max_date = self.missing[idx] + relativedelta(months=1)
        diff = max_date - self.missing[idx]
        dates = [str(self.missing[idx] + dt.timedelta(days=x)).replace('-', '') for x in range(diff.days)]
        return [glob.glob(os.path.join(self.tmp_dir, '*'+x+'*.h5'))[0] for x in dates]

    def standardize_format(self, template, read_smap_loc='/mnt/e/PycharmProjects/thesis/paper2/fetch/read_smap_products.R'):
        
        try:
            read_smap_loc = self.kwargs['read_smap_loc']
        except KeyError as e:
            print('{}\nPlease init this object with the read_smap_loc keyword argument that points to the read_smap_products.R function'.format(e))
            raise
        # Open downloaded dataset and remove empty dimension
        for idx, date in enumerate(self.missing):
            print(date)
            arr_list = []
            img_list = self.get_monthly_names(idx)
            for img in img_list:
                try:
                    dat = h5py.File(img, 'r')
                    arr_list.append(dat['GPP']['gpp_mean'].value)
                except OSError as e:
                    print(e)
                    print('{} corrupted. Skipping to create monthly mean'.format(img))

            mean = np.expand_dims(np.mean(arr_list, axis=0), axis=0)
            tmp_name = os.path.join(self.tmp_dir, str(date).replace('-', '') + '_gpp.tif')
            out_name = os.path.join(self.data_dir, str(date).replace('-', '') + '_gpp.tif')
            transform = rio.transform.from_bounds(-17367530.45, -7314540.83, 17367530.45, 7314540.83, 3856, 1624)

            profile = {'driver': 'GTiff', 'height': 1624, 'width': 3856,
                       'dtype': mean.dtype, 'transform': transform, 'count': 1}

            with rio.open(Path(tmp_name), 'w', crs='EPSG:6933', **profile) as dst:
                dst.write(mean)

            self.tmp_files.append(tmp_name)

            sp.call(['Rscript', '--vanilla', read_smap_loc, tmp_name, template, out_name])
