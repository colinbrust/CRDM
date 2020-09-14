import numpy as np
import rasterio as rio
import os
import glob
import argparse

def to_memmap(f: str, out_dir: str) -> None:
    """
    f: path to geotiff that will be read and written out as a numpy memmap
    out_dir: path to directory that will contain new memmap
    """
    print(f)

    # Read array, convert na values to np.nan, normalize between zero and one
    arr = rio.open(f).read(1)
    arr = np.where(arr <= -9999, np.nan, arr)
    arr = np.float32(np.interp(arr, (np.nanmin(arr), np.nanmax(arr)), (0, +1)))

    # Make empty memmap
    mm = np.memmap(os.path.join(out_dir, os.path.basename(f)).replace('.tif', '.dat'), dtype='float32', mode='w+', shape=arr.shape)
    # Copy .tif array to the memmap.
    mm[:] = arr[:]

    # Flush to disk.
    del mm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert all files in a directory to numpy memmap file')
    parser.add_argument('-d', '--file_dir', type=str, help='Directory .tif images.')
    parser.add_argument('-o', '--out_dir', type=str, help='Directory to write memmap arrays to.')
    args = parser.parse_args()

    f_list = glob.glob(os.path.join(args.file_dir, '*.tif'))

    for f in f_list:
        to_memmap(f=f, out_dir=args.out_dir)
