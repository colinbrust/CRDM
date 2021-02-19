import argparse
from dc.fetch.DownloadFile import download_file
import os
import datetime as dt
from pathlib import Path

# example_url https://n5eil01u.ecs.nsidc.org/SMAP/SPL4SMAU.004/2015.03.31/SMAP_L4_SM_aup_20150331T120000_Vv4030_001.h5

# Make sure to run the following in the terminal so you can properly download data
# echo "machine urs.earthdata.nasa.gov login <uid> password <password>" >> ~/.netrc
# chmod 0600 ~/.netrc


def fetch_l4sm(download_dir, dataset):

    os.chdir(download_dir)
    f_list = [os.path.basename(str(x)) for x in Path(download_dir).glob('*.h5')]

    if dataset == 'l4sm':
        dates = [dt.datetime.strptime(x, 'SMAP_L4_SM_aup_%Y%m%dT120000_Vv4030_001.h5').date() for x in f_list]
    elif dataset == 'l4c':
        dates = [dt.datetime.strptime(x, 'SMAP_L4_C_mdl_%Y%m%dT120000_Vv4030_001.h5').date() for x in f_list]
    else:
        raise ValueError("Dataset argument must be one of 'l4sm' or 'l4c'.")

    min_date = min(dates) if dates else dt.date(2015, 3, 30)
    days = [str(dt.date(2020, 1, 1) - dt.timedelta(days=x)).replace('-', '.') for x in range(1, (dt.date(2020, 1, 1) - min_date).days)]

    for day in days:
        # Make valid URL to download from
        day_stripped = day.replace('.', '')
        if dataset == 'l4sm':
            url = 'https://n5eil01u.ecs.nsidc.org/SMAP/SPL4SMAU.004/{}/SMAP_L4_SM_aup_{}T120000_Vv4030_001.h5'.format(day, day_stripped)
        elif dataset == 'l4c':
            url = 'https://n5eil01u.ecs.nsidc.org/SMAP/SPL4CMDL.004/{}/SMAP_L4_C_mdl_{}T000000_Vv4040_001.h5'.format(day, day_stripped)
        else:
            raise ValueError("Dataset argument must be one of 'l4sm' or 'l4c'.")

        download_file(url)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Download SMAP Data')
    parser.add_argument('-od', '--out_dir', type=str, default='.', help='directory to write data to.')
    parser.add_argument('--l4c', dest='l4c', action='store_true', help='Whether to download L4C or L4SM data.')
    parser.add_argument('--no-l4c', dest='l4c', action='store_false', help='Whether to download L4C or L4SM data.')
    parser.set_defaults(l4c=True)

    args = parser.parse_args()

    fetch_l4sm(args.out_dir, dataset=args.l4c)


# # from FetchMissing import FetchMissing
# class FetchL4SM(FetchMissing):
#
#     def fetch_data(self):
#         os.chdir(self.tmp_dir)
#
#         # Get list of all missing days
#         max_date = (self.missing[0] + relativedelta(months=1)) - dt.timedelta(days=1)
#         min_date = self.missing[-1]
#         days = [str(max_date - dt.timedelta(days=x)).replace('-', '.') for x in range((max_date - min_date).days + 1)]
#
#         for day in days:
#
#             # Make valid URL to download from
#             day_stripped = day.replace('.', '')
#             url = 'https://n5eil01u.ecs.nsidc.org/SMAP/SPL4SMAU.004/{}/SMAP_L4_SM_aup_{}T120000_Vv4030_001.h5'.format(day, day_stripped)
#             f_name = os.path.join(self.tmp_dir, os.path.basename(url))
#             self.tmp_files.append(f_name)
#             # If the file isn't already downloaded, download the file
#             if not os.path.exists(f_name):
#                 self.download_file(url)
#
#     def get_monthly_names(self, idx):
#         max_date = self.missing[idx] + relativedelta(months=1)
#         diff = max_date - self.missing[idx]
#         dates = [str(self.missing[idx] + dt.timedelta(days=x)).replace('-', '') for x in range(diff.days)]
#         return [glob.glob(os.path.join(self.tmp_dir, '*'+x+'*.h5'))[0] for x in dates]
#
#     def standardize_format(self, template):
#
#         try:
#             read_smap_loc = self.kwargs['read_smap_loc']
#         except KeyError as e:
#             print('{}\nPlease init this object with the read_smap_loc keyword argument that points to the read_smap_products.R function'.format(e))
#             raise
#         # Open downloaded dataset and remove empty dimension
#         for idx, date in enumerate(self.missing):
#             print(date)
#             surf_list = []
#             rz_list = []
#             img_list = self.get_monthly_names(idx)
#             for img in img_list:
#                 try:
#                     dat = h5py.File(img, 'r')
#                     surf_list.append(dat['Analysis_Data']['sm_surface_analysis'].value)
#                     rz_list.append(dat['Analysis_Data']['sm_rootzone_analysis'].value)
#
#                 except OSError as e:
#                     print(e)
#                     print('{} corrupted. Skipping to create monthly mean'.format(img))
#
#             surf_mean = np.expand_dims(np.mean(surf_list, axis=0), axis=0)
#             rz_mean = np.expand_dims(np.mean(rz_list, axis=0), axis=0)
#
#             surf_tmp_name = os.path.join(self.tmp_dir, str(date).replace('-', '') + '_sm-surface.tif')
#             surf_out_name = os.path.join(self.data_dir, str(date).replace('-', '') + '_sm-surface.tif')
#
#             rz_tmp_name = os.path.join(self.tmp_dir, str(date).replace('-', '') + '_sm-rootzone.tif')
#             rz_out_name = os.path.join(self.data_dir, str(date).replace('-', '') + '_sm-rootzone.tif')
#
#             transform = rio.transform.from_bounds(-17367530.45, -7314540.83, 17367530.45, 7314540.83, 3856, 1624)
#
#             profile = {'driver': 'GTiff', 'height': 1624, 'width': 3856,
#                        'dtype': surf_mean.dtype, 'transform': transform, 'count': 1}
#
#             with rio.open(Path(surf_tmp_name), 'w', crs='EPSG:6933', **profile) as dst:
#                 dst.write(surf_mean)
#
#             with rio.open(Path(rz_tmp_name), 'w', crs='EPSG:6933', **profile) as dst:
#                 dst.write(rz_mean)
#
#             self.tmp_files.append(surf_tmp_name)
#             self.tmp_files.append(rz_tmp_name)
#
#             sp.call(['Rscript', '--vanilla', read_smap_loc, surf_tmp_name, template, surf_out_name])
#             sp.call(['Rscript', '--vanilla', read_smap_loc, rz_tmp_name, template, rz_out_name])
#
