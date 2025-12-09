import argparse
import json
from typing import Any

import giatools
import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import disk


def image_astype(img: giatools.Image, dtype: np.dtype) -> giatools.Image:
    return giatools.Image(
        data=img.data.astype(dtype),
        axes=img.axes,
        original_axes=img.original_axes,
        metadata=img.metadata,
    )


filters = {
    'gaussian': lambda img, sigma, order=0, axis=None: (
        apply_2d_filter(
            ndi.gaussian_filter,
            img if order == 0 else image_astype(img, float),
            sigma=sigma,
            order=order,
            axes=axis,
        )
    ),
    'median': lambda img, radius: (
        apply_2d_filter(ndi.median_filter, img, footprint=disk(radius))
    ),
    'prewitt_h': lambda img: (
        apply_2d_filter(ndi.prewitt, img, axis=1)
    ),
    'prewitt_v': lambda img: (
        apply_2d_filter(ndi.prewitt, img, axis=0)
    ),
    'sobel_h': lambda img: (
        apply_2d_filter(ndi.sobel, img, axis=1)
    ),
    'sobel_v': lambda img: (
        apply_2d_filter(ndi.sobel, img, axis=0)
    ),
}


def apply_2d_filter(
    filter_impl: callable,
    img: giatools.Image,
    **kwargs: Any,
) -> giatools.Image:
    """
    Apply the 2-D filter to the 2-D/3-D, potentially multi-frame and multi-channel image.
    """
    result_data = None
    for qtc in np.ndindex(
        img.data.shape[ 0],  # Q axis
        img.data.shape[ 1],  # T axis
        img.data.shape[ 2],  # Z axis
        img.data.shape[-1],  # C axis
    ):
        sl = np.s_[*qtc[:3], ..., qtc[3]]
        arr = img.data[sl]
        assert arr.ndim == 2  # sanity check, should always be True

        # Perform 2-D filtering
        res = filter_impl(arr, **kwargs)
        if result_data is None:
            result_data = np.empty(img.data.shape, res.dtype)
        result_data[sl] = res

    # Return results
    return giatools.Image(result_data, img.axes)


def apply_filter(
    input_filepath: str,
    output_filepath: str,
    filter_type: str,
    **kwargs: Any,
):
    # Read the input image
    img = giatools.Image.read(input_filepath)

    # Perform filtering
    filter_impl = filters[filter_type]
    res = filter_impl(img, **kwargs).normalize_axes_like(img.original_axes)

    # Adopt metadata and write the result
    res.metadata = img.metadata
    res.write(output_filepath, backend='tifffile')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input image filepath')
    parser.add_argument('output', type=str, help='Output image filepath (TIFF)')
    parser.add_argument('params', type=str)
    args = parser.parse_args()

    # Read the config file
    with open(args.params) as cfgf:
        cfg = json.load(cfgf)

    apply_filter(
        args.input,
        args.output,
        **cfg,
    )
