import argparse
import json
from typing import Any

import giatools
import numpy as np
import skimage.filters


filters = {
    'frangi': lambda img, **kwargs: (
        apply_nd_filter(skimage.filters.frangi, img, **kwargs)
    ),
    'hessian': lambda img, **kwargs: (
        apply_nd_filter(skimage.filters.hessian, img, **kwargs)
    ),
    'laplace': lambda img, **kwargs: (
        apply_nd_filter(skimage.filters.laplace, img, **kwargs)
    ),
    'meijering': lambda img, **kwargs: (
        apply_nd_filter(skimage.filters.meijering, img, **kwargs)
    ),
    'sato': lambda img, **kwargs: (
        apply_nd_filter(skimage.filters.sato, img, **kwargs)
    ),
}


def apply_nd_filter(
    filter_impl: callable,
    img: giatools.Image,
    dtype: str,
    **kwargs: Any,
) -> giatools.Image:
    """
    Apply the filter to the 2-D/3-D, potentially multi-frame and multi-channel image.
    """
    result_data = np.empty(img.data.shape, dtype=dtype)
    for qtc in np.ndindex(
        img.data.shape[ 0],  # Q axis
        img.data.shape[ 1],  # T axis
        img.data.shape[-1],  # C axis
    ):
        sl = np.s_[*qtc[:2], ..., qtc[2]]  # noqa: E999
        arr = img.data[sl]
        assert arr.ndim == 3  # sanity check, should always be True

        # Perform 2-D or 3-D filtering
        if arr.shape[0] == 1:
            info = 'Performing 2-D filtering'
            result_data[sl][0] = filter_impl(arr[0], **kwargs).astype(dtype)
        else:
            info = 'Performing 3-D filtering'
            result_data[sl] = filter_impl(arr, **kwargs).astype(dtype)

    # Print status info
    print(info)

    # Return results as 16bit, 32bit, or 64bit floating point
    return giatools.Image(result_data.astype(dtype), img.axes)


def apply_filter(
    input_filepath: str,
    output_filepath: str,
    filter_type: str,
    **kwargs: Any,
):
    # Validate and transform input parameters
    params = dict(kwargs)
    if (sigma_min := params.pop('sigma_min', None)) is not None and (sigma_max := params.pop('sigma_max', None)) is not None:
        num_sigma = params.pop('num_sigma')
        if sigma_min < sigma_max:
            params['sigmas'] = np.linspace(sigma_min, sigma_max, num_sigma)
        elif sigma_min == sigma_max:
            params['sigmas'] = [sigma_min]
        else:
            raise ValueError(f'Minimum sigma ({sigma_min:g}) must not be greater than Maximum sigma ({sigma_max:g})')

    # Read the input image
    img = giatools.Image.read(input_filepath)

    # Perform filtering
    print(f'Applying filter: "{filter_type}"')
    filter_impl = filters[filter_type]
    res = filter_impl(img, **params).normalize_axes_like(img.original_axes)

    # Adopt metadata and write the result
    res.metadata = img.metadata
    res.write(output_filepath, backend='tifffile')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
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
