import argparse
import json
from typing import (
    Any,
    Callable,
)

import giatools
import numpy as np
import scipy.ndimage as ndi


def image_astype(image: giatools.Image, dtype: np.dtype) -> giatools.Image:
    if np.issubdtype(image.data.dtype, dtype):
        return image  # no conversion needed
    else:
        return giatools.Image(
            data=image.data.astype(dtype),
            axes=image.axes,
            original_axes=image.original_axes,
            metadata=image.metadata,
        )


def get_anisotropy(image: giatools.Image, axes: str) -> tuple[float, ...] | None:
    """
    Get the anisotropy of the image pixels/voxels along the given `axes`.
    """

    # Determine the pixel/voxel size
    voxel_size = list()
    for axis in axes:
        match axis:
            case 'X':
                if image.metadata.pixel_size is None:
                    return None  # unknown size
                else:
                    voxel_size.append(image.metadata.pixel_size[0])
            case 'Y':
                if image.metadata.pixel_size is None:
                    return None  # unknown size
                else:
                    voxel_size.append(image.metadata.pixel_size[1])
            case 'Z':
                if image.metadata.z_spacing is None:
                    return None  # unknown size
                else:
                    voxel_size.append(image.metadata.z_spacing)

    # Check for unknown size and compute anisotropy
    if any(abs(s) < 1e-8 for s in voxel_size):
        print('Unknown pixel/voxel size')
        return None
    else:
        denom = pow(np.prod(voxel_size), 1 / len(voxel_size))  # geometric mean
        anisotropy = tuple(np.divide(voxel_size, denom).tolist())
        print(f'Anisotropy of {axes} pixels/voxels:', anisotropy)
        return anisotropy


def get_anisotropic_size(image: giatools.Image, axes: str, size: int) -> tuple[int, ...] | int:
    if (anisotropy := get_anisotropy(image, axes)) is not None:
        return tuple(
            np.divide(size, anisotropy).round().clip(1, np.inf).astype(int).tolist(),
        )
    else:
        return size


class Filters:

    @staticmethod
    def gaussian(
        image: giatools.Image,
        sigma: float,
        anisotropic: bool,
        axes: str,
        order: int = 0,
        direction: int | None = None,
        **kwargs: Any,
    ) -> giatools.Image:
        if direction is None:
            _order = 0
        elif order >= 1:
            _order = [0] * len(axes)
            _order[direction] = order
            _order = tuple(_order)
        if anisotropic and (anisotropy := get_anisotropy(image, axes)) is not None:
            _sigma = tuple(np.divide(sigma, anisotropy).tolist())
        else:
            _sigma = sigma
        return apply_nd_filter(
            ndi.gaussian_filter,
            image,
            sigma=_sigma,
            order=_order,
            axes=axes,
            **kwargs,
        )

    @staticmethod
    def uniform(image: giatools.Image, size: int, anisotropic: bool, axes: str, **kwargs: Any) -> giatools.Image:
        _size = get_anisotropic_size(image, axes, size) if anisotropic else size
        return apply_nd_filter(
            ndi.uniform_filter,
            image,
            size=_size,
            axes=axes,
            **kwargs,
        )

    @staticmethod
    def median(image: giatools.Image, size: int, anisotropic: bool, axes: str, **kwargs: Any) -> giatools.Image:
        _size = get_anisotropic_size(image, axes, size) if anisotropic else size
        return apply_nd_filter(
            ndi.median_filter,
            image,
            size=_size,
            axes=axes,
            **kwargs,
        )

    @staticmethod
    def prewitt(image: giatools.Image, direction: int, **kwargs: Any) -> giatools.Image:
        return apply_nd_filter(
            ndi.prewitt,
            image,
            axis=direction,
            **kwargs,
        )

    @staticmethod
    def sobel(image: giatools.Image, direction: int, **kwargs: Any) -> giatools.Image:
        return apply_nd_filter(
            ndi.sobel,
            image,
            axis=direction,
            **kwargs,
        )


def apply_nd_filter(
    filter_impl: Callable[[np.ndarray, Any, ...], np.ndarray],
    image: giatools.Image,
    axes: str,
    **kwargs: Any,
) -> giatools.Image:
    """
    Apply the filter to the 2-D/3-D, potentially multi-frame and multi-channel image.
    """
    print(
        'Applying filter:',
        filter_impl.__name__,
        'with',
        ', '.join(
            f'{key}={repr(value)}' for key, value in (kwargs | dict(axes=axes)).items()
            if not isinstance(value, np.ndarray)
        ),
    )
    result_data = None
    for section_sel, section_arr in image.iterate_jointly(axes):
        assert len(axes) == section_arr.ndim and section_arr.ndim in (2, 3)  # sanity check, always True

        # Define the section using the requested axes layout (compatible with `kwargs`)
        joint_axes_original_order = ''.join(filter(lambda axis: axis in axes, image.axes))
        section = giatools.Image(section_arr, joint_axes_original_order).reorder_axes_like(axes)

        # Perform 2-D or 3-D filtering
        section_result = giatools.Image(
            filter_impl(section.data, **kwargs),
            axes,  # axes layout compatible with `kwargs`
        ).reorder_axes_like(
            joint_axes_original_order,  # axes order compatible to the input `image`
        )

        # Update the result image for the current section
        if result_data is None:
            result_data = np.empty(image.data.shape, section_result.data.dtype)
        result_data[section_sel] = section_result.data

    # Return results
    return giatools.Image(result_data, image.axes, image.metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input image filepath')
    parser.add_argument('output', type=str, help='Output image filepath (TIFF)')
    parser.add_argument('params', type=str)
    args = parser.parse_args()

    # Read the config file
    with open(args.params) as cfgf:
        cfg = json.load(cfgf)
    cfg.setdefault('axes', 'YX')

    # Read the input image
    image = giatools.Image.read(args.input)
    print('Input image shape:', image.data.shape)
    print('Input image axes:', image.axes)
    print('Input image dtype:', image.data.dtype)

    # Convert the image to the explicitly requested `dtype`, or the same as the input image
    convert_to = getattr(np, cfg.pop('dtype', str(image.data.dtype)))
    if np.issubdtype(image.data.dtype, convert_to):
        convert_to = image.data.dtype  # use the input image `dtype` if `convert_to` is a superset
    elif convert_to == np.floating:
        convert_to = np.float64  # use `float64` if conversion to *any* float is *required*

    # If the input image is `float16` or conversion to `float16` is requested, ...
    if convert_to == np.float16:
        image = image_astype(image, np.float32)  # ...convert to `float32` as an intermediate
    else:
        image = image_astype(image, convert_to)  # ...otherwise, convert now

    # Perform filtering
    filter_type = cfg.pop('filter_type')
    filter_impl = getattr(Filters, filter_type)
    result = filter_impl(image, **cfg)

    # Apply `dtype` conversion
    result = image_astype(result, convert_to)

    # Write the result
    result = result.normalize_axes_like(
        image.original_axes,
    )
    print('Output image shape:', result.data.shape)
    print('Output image axes:', result.axes)
    print('Output image dtype:', result.data.dtype)
    result.write(args.output, backend='tifffile')
