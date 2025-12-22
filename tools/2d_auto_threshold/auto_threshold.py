"""
Copyright 2017-2025 Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
"""

import argparse
import json

import giatools
import numpy as np
import skimage.filters
import skimage.util

# Fail early if an optional backend is not available
giatools.require_backend('omezarr')


class DefaultThresholdingMethod:

    def __init__(self, thres, **kwargs):
        self.thres = thres
        self.kwargs = kwargs

    def __call__(self, image, *args, offset=0, **kwargs):
        thres = self.thres(image, *args, **(self.kwargs | kwargs))
        return image > thres + offset

    def __str__(self):
        return self.thres.__name__


class ManualThresholding:

    def __call__(self, image, threshold1: float, threshold2: float | None, **kwargs):
        if threshold2 is None:
            return image > threshold1
        else:
            thres1, thres2 = sorted((threshold1, threshold2))
            return skimage.filters.apply_hysteresis_threshold(image, threshold1, threshold2)

    def __str__(self):
        return 'Manual'


methods = {
    'manual': ManualThresholding(),

    'otsu': DefaultThresholdingMethod(skimage.filters.threshold_otsu),
    'li': DefaultThresholdingMethod(skimage.filters.threshold_li),
    'yen': DefaultThresholdingMethod(skimage.filters.threshold_yen),
    'isodata': DefaultThresholdingMethod(skimage.filters.threshold_isodata),

    'loc_gaussian': DefaultThresholdingMethod(skimage.filters.threshold_local, method='gaussian'),
    'loc_median': DefaultThresholdingMethod(skimage.filters.threshold_local, method='median'),
    'loc_mean': DefaultThresholdingMethod(skimage.filters.threshold_local, method='mean'),
}


def do_thresholding(
    input_filepath: str,
    output_filepath: str,
    method: str,
    invert: bool,
    **kwargs,
):
    assert method in methods, f'Unknown method "{method}"'

    # Read the input image
    img_in = giatools.Image.read(args.input)
    print('Input image shape:', img_in.data.shape)
    print('Input image axes:', img_in.axes)
    print('Input image dtype:', img_in.data.dtype)

    # Perform thresholding
    method_impl = methods[method]
    print(
        'Thresholding:',
        str(method_impl),
        'with',
        ', '.join(
            f'{key}={repr(value)}' for key, value in (kwargs | dict(invert=invert)).items()
        ),
    )
    result = np.empty(img_in.data.shape, bool)
    for sl, section in img_in.iterate_jointly('ZYX'):
        result[sl] = method_impl(
            image=np.asarray(section),  # some implementations have issues with Dask arrays
            **kwargs,
        )
    if invert:
        result = np.logical_not(result)

    # Convert to canonical representation for binary images
    result = (result * 255).astype(np.uint8)

    # Write result
    img_out = giatools.Image(
        data=skimage.util.img_as_ubyte(result),
        axes=img_in.axes,
    ).normalize_axes_like(
        img_in.original_axes,
    )
    print('Output image shape:', img_out.data.shape)
    print('Output image axes:', img_out.axes)
    print('Output image dtype:', img_out.data.dtype)
    img_out.write(
        output_filepath,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automatic image thresholding')
    parser.add_argument('input', type=str, help='Path to the input image')
    parser.add_argument('output', type=str, help='Path to the output image (uint8)')
    parser.add_argument('params', type=str)
    args = parser.parse_args()

    # Read the config file
    with open(args.params) as cfgf:
        cfg = json.load(cfgf)

    # Perform the thresholding
    do_thresholding(
        args.input,
        args.output,
        **cfg,
    )
