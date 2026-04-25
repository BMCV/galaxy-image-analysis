import argparse
import sys

import giatools.io
import numpy as np
import skimage.exposure
import skimage.io
import skimage.util


def rescale(img):
    assert np.issubdtype(img.dtype, np.floating), str(img.dtype)  # sanity check
    img = img - img.min()
    return img / img.max() if img.max() != 0 else 1


hOptions = {
    'default': lambda img_raw: rescale(skimage.exposure.equalize_hist(img_raw)),  # rescale needed for values to be in [0, 1]
    'clahe': lambda img_raw: skimage.exposure.equalize_adapthist(img_raw),  # produces image with values in [0, 1]
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Histogram equalization')
    parser.add_argument('input_file', type=argparse.FileType('r'), default=sys.stdin, help='input file')
    parser.add_argument('out_file', type=argparse.FileType('w'), default=sys.stdin, help='out file (TIFF)')
    parser.add_argument('h_type', choices=hOptions.keys(), help='histogram equalization method')
    args = parser.parse_args()

    img_in = giatools.io.imread(args.input_file.name)
    res = hOptions[args.h_type](img_in)
    res = skimage.util.img_as_uint(res)
    skimage.io.imsave(args.out_file.name, res, plugin="tifffile")
