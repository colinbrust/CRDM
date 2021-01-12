from crdm.fetch.DownloadFile import download_file
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
    for v in ['tmmx', 'tmmn', 'pr', 'vs', 'rmax', 'rmin', 'srad', 'vpd']:
        fetch_gridmet('/mnt/e/PycharmProjects/CRDM/data/raw/gridmet/ncdf', v)


# class FetchGridmet(FetchMissing):
#
#     def fetch_data(self):
#         os.chdir(self.download_dir)
#
#         # Gridmet data location
#         url = 'http://www.northwestknowledge.net/metdata/data/'
#
#         # Because Gridmet data are stacked as one NetCDF per year, we make a list of urls with all years that we have
#         # missing data
#         missing_years = set([x.year for x in self.missing])
#         urls = [url + self.variable + '_' + str(x) + '.nc' for x in missing_years]
#
#         # Download gridmet data for every year that is missing using a command line call to wget
#         # If the file isn't already downloaded, download the file
#
#         for url in urls:
#             if not os.path.exists(os.path.basename(url)):
#                 self.download_file(url)
#
#         # Keep track of the files that we downloaded
#         self.f_names = [os.path.join(self.tmp_dir, os.path.basename(x)) for x in urls]
#         self.tmp_files += self.f_names
#
#     def standardize_format(self, template):
#
#         for f in self.f_names:
#             # Open downloaded dataset and remove empty dimension
#             dat = xarray.open_dataset(f)
#             arr = dat.squeeze().to_array().squeeze()
#
#             # Data are stacked in (n x lat x lon) arrays where n is either 365 or the number of days passed in the current
#             # year so far. Make a list of indices corresponding to months so that we can properly slice to get monthly
#             # averages/totals.
#             date_indices = np.cumsum([calendar.monthrange(x.year, x.month)[1] for x in reversed(self.missing)]).tolist()
#             date_indices.insert(0, 0)
#             date_labs = [str(x).replace('-', '') for x in reversed(self.missing)]
#
#             for mon in range(len(date_indices) - 1):
#                 print(date_labs[mon])
#                 # If it is precipitation, take the sum of all images in the month, otherwise take the mean
#                 if self.variable == 'pr':
#                     tmp = np.sum(arr[date_indices[mon]:date_indices[mon + 1], :, :], 0)
#                 else:
#                     tmp = np.mean(arr[date_indices[mon]:date_indices[mon + 1], :, :], 0)
#
#                 base_name = date_labs[mon] + '_' + self.variable + '.tif'
#
#                 # Force CRS to 4326 (should already be this when downloading from Gridmet server) and write a temporary file.
#                 tmp.rio.set_crs('epsg:4326')
#                 tmp.rio.to_raster(os.path.join(self.tmp_dir, base_name))
#                 self.tmp_files.append(os.path.join(self.tmp_dir, base_name))
#
#                 # Reproject the image to match our template (9km) file so all data are in a standardized format.
#                 reproject_image_to_template(template=template, newDs=os.path.join(self.tmp_dir, base_name),
#                                             out_name=os.path.join(self.data_dir, base_name))
