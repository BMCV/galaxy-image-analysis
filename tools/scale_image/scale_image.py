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
    factor: float,
) -> tuple[float, ...]:
    """
    Determine a tuple of `scale` factors for uniform or spatially uniform scaling.

    Axes, that are not present in the original image data, are ignored.
    """
    ignored_axes = [
        axis for axis_idx, axis in enumerate(img.axes)
        if axis not in img.original_axes or (
            factor < 1 and img.data.shape[axis_idx] == 1
        )
    ]
    match axes:

        case 'all':
            return tuple(
                [
                    (factor if axis not in ignored_axes else 1)
                    for axis in img.axes if axis != 'C'
                ]
            )

        case 'spatial':
            return tuple(
                [
                    (factor if axis in 'YXZ' and axis not in ignored_axes else 1)
                    for axis in img.axes if axis != 'C'
                ]
            )

        case '_':
            raise ValueError(f'Unknown axes for uniform scaling: "{axes}"')


def get_scale_for_isotropy(
    img: giatools.Image,
    sample: Literal['up', 'down'],
) -> tuple[float, ...]:
    """
    Determine a tuple of `scale` factors to establish spatial isotropy.

    The `sample` parameter governs whether to up-sample or down-sample the image data.
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
        match sample:
            case 'up':
                return voxel_size.max() / voxel_size
            case 'down':
                return voxel_size.min() / voxel_size
            case '_':
                raise ValueError(f'Unknown value for sample: "{sample}"')

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
    new_pixel_size = np.divide(
        old_pixel_size,
        (
            scales[old.axes.index('X')],
            scales[old.axes.index('Y')],
        ),
    )
    metadata['resolution'] = tuple(1 / new_pixel_size)

    # Update the metadata for the new voxel depth
    old_voxel_depth = old.metadata.get('z_spacing', 1)
    metadata['z_spacing'] = old_voxel_depth / scales[old.axes.index('Z')]

    return metadata


def metadata_to_str(metadata: dict) -> str:
    tokens = list()
    for key in sorted(metadata.keys()):
        value = metadata[key]
        if isinstance(value, tuple):
            value = '(' + ', '.join([f'{val}' for val in value]) + ')'
        tokens.append(f'{key}: {value}')
    if len(metadata_str := ', '.join(tokens)) > 0:
        return metadata_str
    else:
        return 'has no metadata'


def write_output(filepath: str, img: giatools.Image):
    """
    Validate that the output file format is suitable for the image data, then write it.
    """
    print('Output shape:', img.data.shape)
    print('Output axes:', img.axes)
    print('Output', metadata_to_str(img.metadata))

    # Validate that the output file format is suitable for the image data
    if filepath.lower().endswith('.png'):
        if not frozenset(img.axes) <= frozenset('YXC'):
            sys.exit(f'Cannot write PNG file with axes "{img.axes}"')

    # Write image data to the output file
    img.write(filepath)


def scale_image(
    input_filepath: str,
    output_filepath: str,
    mode: Literal['uniform', 'explicit', 'isotropy'],
    order: int,
    anti_alias: bool,
    **cfg,
):
    img = giatools.Image.read(input_filepath)
    print('Input axes:', img.original_axes)
    print('Input', metadata_to_str(img.metadata))

    # Determine `scale` for scaling
    match mode:

        case 'uniform':
            scale = get_uniform_scale(img, cfg['axes'], cfg['factor'])

        case 'explicit':
            scale = tuple(
                [cfg.get(f'factor_{axis.lower()}', 1) for axis in img.axes if axis != 'C']
            )

        case 'isotropy':
            scale = get_scale_for_isotropy(img, cfg['sample'])

        case '_':
            raise ValueError(f'Unknown mode: "{mode}"')

    # Assemble remaining `rescale` parameters
    rescale_kwargs = dict(
        scale=scale,
        order=order,
        preserve_range=True,
        channel_axis=img.axes.index('C'),
    )
    if (anti_alias := anti_alias and (np.array(scale) < 1).any()):
        rescale_kwargs['anti_aliasing'] = anti_alias
        rescale_kwargs['anti_aliasing_sigma'] = tuple(
            [
                get_aa_sigma_by_scale(s) for s in scale
            ] + [0]  # `skimage.transform.rescale` also expects a value for the channel axis
        )
    else:
        rescale_kwargs['anti_aliasing'] = False

    # Re-sample the image data to perform the scaling
    for key, value in rescale_kwargs.items():
        print(f'{key}: {value}')
    arr = skimage.transform.rescale(img.data, **rescale_kwargs)

    # Preserve the `dtype` so that both brightness and range of values is preserved
    if arr.dtype != img.data.dtype:
        if np.issubdtype(img.data.dtype, np.integer):
            arr = arr.round()
        arr = arr.astype(img.data.dtype)

    # Determine the result metadata and save result
    metadata = get_new_metadata(img, scale, arr)
    write_output(
        output_filepath,
        giatools.Image(
            data=arr,
            axes=img.axes,
            metadata=metadata,
        ).squeeze()
    )


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
