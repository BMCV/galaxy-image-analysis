"""
Copyright 2017-2024 Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
"""

import argparse

import numpy as np
import skimage.filters
import skimage.io
import skimage.util
import tifffile

th_methods = {
    'manual': lambda thres, **kwargs: thres,

    'otsu': lambda img_raw, **kwargs: skimage.filters.threshold_otsu(img_raw),
    'li': lambda img_raw, **kwargs: skimage.filters.threshold_li(img_raw),
    'yen': lambda img_raw, **kwargs: skimage.filters.threshold_yen(img_raw),
    'isodata': lambda img_raw, **kwargs: skimage.filters.threshold_isodata(img_raw),

    'loc_gaussian': lambda img_raw, bz, **kwargs: skimage.filters.threshold_local(img_raw, bz, method='gaussian'),
    'loc_median': lambda img_raw, bz, **kwargs: skimage.filters.threshold_local(img_raw, bz, method='median'),
    'loc_mean': lambda img_raw, bz, **kwargs: skimage.filters.threshold_local(img_raw, bz, method='mean')
}


def do_thresholding(in_fn, out_fn, th_method, block_size=5, threshold=0, invert_output=False):
    img = skimage.io.imread(in_fn)
    th = th_methods[th_method](img_raw=img, bz=block_size, thres=threshold)
    res = img > th
    if invert_output:
        res = np.logical_not(res)
    tifffile.imwrite(out_fn, skimage.util.img_as_ubyte(res))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automatic Image Thresholding')
    parser.add_argument('im_in', help='Path to the input image')
    parser.add_argument('im_out', help='Path to the output image (TIFF)')
    parser.add_argument('th_method', choices=th_methods.keys(), help='Thresholding method')
    parser.add_argument('block_size', type=int, default=5, help='Odd size of pixel neighborhood for calculating the threshold')
    parser.add_argument('threshold', type=float, default=0, help='Manual thresholding value')
    parser.add_argument('invert_output', default=False, type=bool, help='Values below/above the threshold are labeled with 0/255 if False, and with 255/0 otherwise')
    args = parser.parse_args()

    do_thresholding(args.im_in, args.im_out, args.th_method, args.block_size, args.threshold, args.invert_output)
