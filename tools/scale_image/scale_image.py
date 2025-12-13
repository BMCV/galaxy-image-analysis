import argparse
import json
import sys
from typing import (
    Any,
    Literal,
)

import giatools.io
import numpy as np
import skimage.io
import skimage.transform
import skimage.util


def get_uniform_scale(
    img: giatools.Image,
    axes: Literal['all', 'spatial'],
) -> float | tuple[float, ...]:
    """
    Determine a `scale` factor or a tuple of `scale` factors for uniform or spatially uniform scaling.
    """
    match axes:

        case 'all':
            return cfg['factor']

        case 'spatial':
            return tuple(
                [
                    (cfg['factor'] if axis in 'YXZ' else 1)
                    for axis in img.axes if axis != 'C'
                ]
            )

        case '_':
            raise ValueError(f'Unknown axes for uniform scaling: "{axes}"')


def get_scale_for_isotropy(
    img: giatools.Image,
    mode: Literal['up', 'down'],
) -> tuple[float, ...]:
    """
    Determine a tuple of `scale` factors to establish spatial isotropy.

    The `mode` parameter governs whether up-sampling or down-sampling will be performed.
    """
    scale = tuple([1] * len(img.axes) - 1)  # omit the channel axis
    z_axis, y_axis, x_axis = [
        img.axes.index(axis) for axis in 'zyx'
    ]

    # Determine the pixel size of the image
    if 'resolution' in img.metadata:
        pixel_size = np.divide(1, img.metadata['resolution'])
    else:
        sys.exit('Resolution information missing in image metadata')

    # Define unified transformation of voxel sizes to scale factors
    def voxel_size_to_scale(voxel_size: np.ndarray) -> np.ndarray:
        match mode:
            case 'up':
                return voxel_size.max() / voxel_size
            case 'down':
                return voxel_size.min() / voxel_size
            case '_':
                raise ValueError(f'Unknown mode: "{mode}"')

    # Handle the 3-D case
    if img.shape[img.axes.index('Z')] > 1:

        # Determine the voxel depth of the image
        if (voxel_depth := img.metadata.get('z_spacing', None)) is None:
            sys.exit('Voxel depth information missing in image metadata')

        # Determine the XYZ scale factors
        scale[x_axis], scale[y_axis], scale[z_axis] = (
            voxel_size_to_scale(
                np.array([*pixel_size, voxel_depth]),
            )
        )

    # Handle the 2-D case
    else:

        # Determine the XYZ scale factors
        scale[x_axis], scale[y_axis] = (
            voxel_size_to_scale(
                np.array(pixel_size),
            )
        )

    return scale


def get_aa_sigma_by_scale(scale: float) -> float:
    """
    Determine the optimal size of the Guassian filter for anti-aliasing.

    See for details: https://scikit-image.org/docs/0.25.x/api/skimage.transform.html#skimage.transform.rescale
    """
    return (1 / scale - 1) / 2 if scale < 1 else 0


def get_new_metadata(
    old: giatools.Image,
    scale: float | tuple[float, ...],
    arr: np.ndarray,
) -> dict[str, Any]:
    """
    Determine the result metadata (copy and adapt).
    """
    metadata = dict(old.metadata)
    scales = (
        [scale] * (len(old.axes) - 1)  # omit the channel axis
        if isinstance(scale, float) else scale
    )

    # Determine the original pixel size
    old_pixel_size = (
        np.divide(1, old.metadata['resolution'])
        if 'resolution' in old.metadata else (1, 1)
    )

    # Determine the new pixel size and update metadata
    new_pixel_size = np.multiply(
        old_pixel_size,
        (
            scales[old.axes.index('X')],
            scales[old.axes.index('Y')],
        ),
    )
    metadata['resolution'] = 1 / new_pixel_size

    # Update the metadata for the new voxel depth
    old_voxel_depth = old.metadata.get('z_spacing', 1)
    metadata['z_spacing'] = old_voxel_depth * scales[old.axes.index('Z')]

    return metadata


def scale_image(
    input_filepath: str,
    output_filepath: str,
    mode: Literal['uniform', 'non-uniform', 'isotropy'],
    order: int,
    anti_alias: bool,
    **cfg,
):
    img = giatools.Image.read(input_filepath)

    # Determine `scale` for scaling
    match mode:

        case 'uniform':
            scale = get_uniform_scale(img, cfg['axes'])

        case 'non-uniform':
            scale = [cfg.get(f'factor_{axis}', 1) for axis in img.axes if axis != 'C']

        case 'isotropy':
            scale = get_scale_for_isotropy(img, cfg['isotropy_mode'])

        case '_':
            raise ValueError(f'Unknown mode: "{mode}"')

    # Assemble remaining `rescale` parameters
    rescale_kwargs = dict(
        scale=scale,
        order=order,
        preserve_range=True,
        channel_axis=img.axes.index('C'),
    )
    if isinstance(scale, float):  # uniform scaling
        if (anti_alias := anti_alias and scale < 1):
            rescale_kwargs['anti_aliasing'] = anti_alias
            rescale_kwargs['anti_aliasing_sigma'] = get_aa_sigma_by_scale(scale)
    else:  # non-uniform scaling
        if (anti_alias := anti_alias and (np.array(scale) < 1).any()):
            rescale_kwargs['anti_aliasing'] = anti_alias
            rescale_kwargs['anti_aliasing_sigma'] = tuple(
                [
                    get_aa_sigma_by_scale(s) for s in scale
                ] + [0]  # `skimage.transform.rescale` also expects a value for the channel axis
            )

    # Re-sample the image data to perform the scaling
    print('-' * 10)
    print(rescale_kwargs)
    print('-' * 10)
    arr = skimage.transform.rescale(img.data, **rescale_kwargs)

    # Preserve the `dtype` so that both brightness and range of values is preserved
    if arr.dtype != img.data.dtype:
        if np.issubdtype(img.data.dtype, np.integer):
            arr = arr.round()
        arr = arr.astype(img.data.dtype)

    # Determine the result metadata and save result
    metadata = get_new_metadata(img, scale, arr)
    giatools.Image(
        data=arr,
        axes=img.axes,
        metadata=metadata,
    ).normalize_axes_like(
        img.original_axes,
    ).write(output_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('params', type=str)
    args = parser.parse_args()

    # Read the config file
    with open(args.params) as cfgf:
        cfg = json.load(cfgf)

    # Perform scaling
    scale_image(
        args.input,
        args.output,
        **cfg,
    )
