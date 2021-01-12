import argparse
import os
from osgeo import gdal
from pathlib import Path


"""
Taken from: https://github.com/jgomezdans/eoldas_ng_observations/blob/master/eoldas_ng_observations/eoldas_observation_helpers.py#L29
"""


def reproject_image_to_template(template, newDs, out_name, method='cubic'):
    newDs_ds = gdal.Open(newDs)
    if newDs_ds is None:
        raise IOError("GDAL could not open newDs file {}".format(newDs))
    newDs_proj = newDs_ds.GetProjection()
    data_type = newDs_ds.GetRasterBand(1).DataType
    n_bands = newDs_ds.RasterCount

    template_ds = gdal.Open(template)
    if template_ds is None:
        raise IOError("GDAL could not open template file {}".format(template))
    template_proj = template_ds.GetProjection()
    template_geotrans = template_ds.GetGeoTransform()
    w = template_ds.RasterXSize
    h = template_ds.RasterYSize

    proj_methods = {'cubic': gdal.GRA_Cubic,
                    'nearest': gdal.GRA_NearestNeighbour}

    try:
        m = proj_methods.pop(method)
    except KeyError as e:
        print("Error: {}. Method argument must be one of 'cubic' or 'nearest'")

    dst_ds = gdal.GetDriverByName('GTiff').Create(out_name, w, h, n_bands, data_type)
    dst_ds.SetGeoTransform(template_geotrans)
    dst_ds.SetProjection(template_proj)
    gdal.ReprojectImage(newDs_ds, dst_ds, newDs_proj, template_proj, m)
    dst_ds = None  # Flush to disk
    return out_name


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Reproject/Resize a directory of GTiffs to a template proj/size')
    parser.add_argument('-t', '--template', type=str, nargs=1,
                        help='Template GTiff file')
    parser.add_argument('-od' '--old_dir', type=str, nargs=1,
                        help='Directory containing images for reprojection')
    parser.add_argument('-nd', '--new_dir', type=str, nargs=1,
                        help='Directory where new images will be saved')
    parser.add_argument('-m', '--method', type=str, nargs=1, default='cubic',
                        help='Resampling method. Can be one of "cubic" or "nearest"')

    args = parser.parse_args()

    for f in Path(args.old_dir[0]).glob('*.tif'):
        new_name = os.path.join(args.new_dir[0], os.path.basename(f))
        reproject_image_to_template(args.template[0], str(f), new_name)


