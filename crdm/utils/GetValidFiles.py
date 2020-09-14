import os
import glob
import re
import shutil

def get_valid_files(f_dir):
    f_list = glob.glob(os.path.join(f_dir, '*.tif'))
    variables = list(set([os.path.basename(x).split('_')[-1].replace('.tif', '') for x in f_list]))

    d_list = []
    for var in variables:
        pat = re.compile(var)
        imgs = list(filter(pat.search, f_list))
        dates = [os.path.basename(x).split('_')[0] for x in imgs]
        d_list.append(dates)
    
    common_dates = set.intersection(*[set(l) for l in d_list])
    return list(common_dates)


def move_valid_files(f_dir, out_dir):

    dates = get_valid_files(f_dir=f_dir)
    f_list = glob.glob(os.path.join(f_dir, '*.tif'))

    for date in dates:
        pat = re.compile(date)
        imgs = list(filter(pat.search, f_list))
        new_imgs = [os.path.join(out_dir, os.path.basename(x)) for x in imgs]

        for old, new in tuple(zip(imgs, new_imgs)):
            shutil.copyfile(old, new)