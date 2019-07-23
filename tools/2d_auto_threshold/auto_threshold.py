import argparse
import numpy as np
import os
import sys 
import warnings
import skimage.io
import skimage.filters
import skimage.util

threshOptions = {
    'otsu' : lambda img_raw: skimage.filters.threshold_otsu(img_raw),
    'gaussian_adaptive' : lambda img_raw: skimage.filters.threshold_local(img_raw, 3, method='gaussian')
    'mean_adaptive' : lambda img_raw: skimage.filters.threshold_local(img_raw, 3, method='mean')
    'isodata' : lambda img_raw: skimage.filters.threshold_isodata(img_raw),
    'li' : lambda img_raw: skimage.filters.threshold_li(img_raw),
    'yen' : lambda img_raw: skimage.filters.threshold_yen(img_raw),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segment Foci')
    parser.add_argument('input_file', type=argparse.FileType('r'), default=sys.stdin, help='input file')
    parser.add_argument('out_file', type=argparse.FileType('w'), default=sys.stdin, help='out file (TIFF)')
    parser.add_argument('thresh_type', choices=threshOptions.keys(), help='thresholding method')
    parser.add_argument('dark_background', default=True, type=bool, help='True if background is dark')
    args = parser.parse_args()

    img_in = skimage.io.imread(args.input_file.name)
    img_in = np.reshape(img_in, [img_in.shape[0], img_in.shape[1]])
    thresh = threshOptions[args.thresh_type](img_in)

    if args.dark_background:
        res = img_in > thresh
    else:
        res = img_in <= thresh

    with warnings.catch_warnings():
    	warnings.simplefilter("ignore")
    	res = skimage.util.img_as_uint(res)
    	skimage.io.imsave(args.out_file.name, res, plugin="tifffile")
