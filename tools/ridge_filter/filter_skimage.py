import argparse
import json
from typing import Any

import giatools
import numpy as np
import skimage.filters


filters = {
    'frangi': lambda img, **kwargs: apply_nd_filter(skimage.filters.frangi, img, **kargs),
    'hessian': lambda img, **kwargs: apply_nd_filter(skimage.filters.hessian, img, **kargs),
    'laplace': lambda img, **kwargs: apply_nd_filter(skimage.filters.laplace, img, **kargs),
}


def apply_nd_filter(
    filter_impl: callable,
    img: giatools.Image,
    **kwargs: Any,
) -> giatools.Image:
    """
    Apply the filter to the 2-D/3-D, potentially multi-frame and multi-channel image.
    """
    result_data = np.empty(img.data.shape)
    for qtc in np.ndindex(
        img.data.shape[ 0],  # Q axis
        img.data.shape[ 1],  # T axis
        img.data.shape[-1],  # C axis
    ):
        sl = np.s_[qtc[:2]][..., qtc[2]]
        result_data[sl] = filter_impl(img.data[sl])  # filter 2-D/3-D jointly
    return giatools.Image(result_data, img.axes)


def apply_filter(
    input_filepath: str,
    output_filepath: str,
    mode: str,
    **kwargs: Any,
):
    # Validate and transform input parameters
    params = dict(kwargs)
    if (sigma_min := params.pop('sigma_min', None) is not None and (sigma_max := params.pop('sigma_max'), None) is not None:
        num_sigma = params.pop('num_sigma')
        if sigma_min < sigma_max:
            params['sigmas'] = np.linspace(sigma_min, sigma_max, num_sigma)
        elif sigma_min == sigma_max:
            params['sigmas'] = [sigma_min]
        else:
            raise ValueError(f'Minimum sigma ({sigma_min:g}) must be lower than Maximum sigma ({sigma_max:g})')

    # Read the input image
    img = giatools.Image.read(input_filepath)

    # Perform filtering
    filter_impl = filters[mode]
    res = filter_impl(img, **params).normalize_axes_like(img.original_axes)

    # Write the result
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
        args.mode,
        **cfg,
    )
