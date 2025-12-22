"""
Copyright 2017-2025 Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
"""

import argparse

import giatools
import numpy as np
import skimage.filters
import skimage.util


class DefaultThresholdingMethod:

    def __init__(self, thres, accept: list[str] | None = None, **kwargs):
        self.thres = thres
        self.accept = accept if accept else []
        self.kwargs = kwargs

    def __call__(self, image, *args, offset=0, **kwargs):
        accepted_kwargs = self.kwargs.copy()
        for key, val in kwargs.items():
            if key in self.accept:
                accepted_kwargs[key] = val
        thres = self.thres(image, *args, **accepted_kwargs)
        return image > thres + offset


class ManualThresholding:

    def __call__(self, image, thres1: float, thres2: float | None, **kwargs):
        if thres2 is None:
            return image > thres1
        else:
            thres1, thres2 = sorted((thres1, thres2))
            return skimage.filters.apply_hysteresis_threshold(image, thres1, thres2)


th_methods = {
    'manual': ManualThresholding(),

    'otsu': DefaultThresholdingMethod(skimage.filters.threshold_otsu),
    'li': DefaultThresholdingMethod(skimage.filters.threshold_li),
    'yen': DefaultThresholdingMethod(skimage.filters.threshold_yen),
    'isodata': DefaultThresholdingMethod(skimage.filters.threshold_isodata),

    'loc_gaussian': DefaultThresholdingMethod(skimage.filters.threshold_local, accept=['block_size'], method='gaussian'),
    'loc_median': DefaultThresholdingMethod(skimage.filters.threshold_local, accept=['block_size'], method='median'),
    'loc_mean': DefaultThresholdingMethod(skimage.filters.threshold_local, accept=['block_size'], method='mean'),
}


def do_thresholding(
    input_filepath: str,
    output_filepath: str,
    th_method: str,
    block_size: int,
    offset: float,
    threshold1: float,
    threshold2: float | None,
    invert_output: bool,
):
    assert th_method in th_methods, f'Unknown method "{th_method}"'

    # Load image
    img_in = giatools.Image.read(input_filepath)

    # Perform thresholding
    result = th_methods[th_method](
        image=img_in.data,
        block_size=block_size,
        offset=offset,
        thres1=threshold1,
        thres2=threshold2,
    )
    if invert_output:
        result = np.logical_not(result)

    # Convert to canonical representation for binary images
    result = (result * 255).astype(np.uint8)

    # Write result
    giatools.Image(
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
    parser.add_argument('block_size', type=int, help='Odd size of pixel neighborhood for calculating the threshold')
    parser.add_argument('offset', type=float, help='Offset of automatically determined threshold value')
    parser.add_argument('threshold1', type=float, help='Manual threshold value')
    parser.add_argument('--threshold2', type=float, help='Second manual threshold value (for hysteresis thresholding)')
    parser.add_argument('--invert_output', default=False, action='store_true', help='Values below/above the threshold are labeled with 0/255 by default, and with 255/0 if this argument is used')
    args = parser.parse_args()

    do_thresholding(
        args.input,
        args.output,
        args.th_method,
        args.block_size,
        args.offset,
        args.threshold1,
        args.threshold2,
        args.invert_output,
    )
