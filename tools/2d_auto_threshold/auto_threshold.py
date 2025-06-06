"""
Copyright 2017-2024 Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
"""

import argparse

import numpy as np
import skimage.filters
import skimage.util
from giatools.image import Image


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


def do_thresholding(
    input_filepath: str,
    output_filepath: str,
    th_method: str,
    block_size: int,
    offset: float,
    threshold: float,
    invert_output: bool = False,
):
    assert th_method in th_methods, f'Unknown method "{th_method}"'

    # Load image
    img_in = Image.read(input_filepath)

    # Perform thresholding
    threshold = offset + th_methods[th_method](
        img_raw=img_in.data,
        bz=block_size,
        thres=threshold,
    )
    result = img_in.data > threshold
    if invert_output:
        result = np.logical_not(result)

    # Write result
    Image(
        data=skimage.util.img_as_ubyte(result),
        axes=img_in.axes,
    ).normalize_axes_like(
        img_in.original_axes,
    ).write(
        output_filepath,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automatic image thresholding')
    parser.add_argument('input', type=str, help='Path to the input image')
    parser.add_argument('output', type=str, help='Path to the output image (uint8)')
    parser.add_argument('th_method', choices=th_methods.keys(), help='Thresholding method')
    parser.add_argument('block_size', type=int, default=5, help='Odd size of pixel neighborhood for calculating the threshold')
    parser.add_argument('offset', type=float, default=0, help='Offset of automatically determined threshold value')
    parser.add_argument('threshold', type=float, default=0, help='Manual threshold value')
    parser.add_argument('--invert_output', default=False, action='store_true', help='Values below/above the threshold are labeled with 0/255 by default, and with 255/0 if this argument is used')
    args = parser.parse_args()

    do_thresholding(
        args.input,
        args.output,
        args.th_method,
        args.block_size,
        args.offset,
        args.threshold,
        args.invert_output,
    )
