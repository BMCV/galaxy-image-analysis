import argparse
from typing import Any

import giatools
import numpy as np
import skimage.io
import skimage.util


def concat_channels(
    input_image_paths: list[str],
    output_image_path: str,
    axis: str,
    preserve_values: bool,
):
    # Create list of arrays to be concatenated
    images = list()
    metadata = dict()
    for image_path in input_image_paths:

        img = giatools.Image.read(image_path, normalize_axes=giatools.default_normalized_axes)
        arr = img.data

        # Preserve values: Convert to `float` dtype without changing the values
        if preserve_values:
            arr = arr.astype(float)

        # Preserve brightness: Scale values to 0..1
        else:
            arr = skimage.util.img_as_float(arr)

        # Record the metadata
        for metadata_key, metadata_value in img.metadata.items():
            metadata.setdefault(metadata_key, list())
            metadata[metadata_key].append(metadata_value)

        # Record the image data
        images.append(arr)

    # Determine consensual metadata
    # TODO: Convert metadata of images with different units of measurement into a common unit
    final_metadata = dict()
    for metadata_key, metadata_values in metadata.items():
        if (metadata_value := reduce_metadata(metadata_values)) is not None:
            final_metadata[metadata_key] = metadata_value

    # Do the concatenation
    axis_pos = giatools.default_normalized_axes.index(axis)
    arr = np.concatenate(images, axis_pos)
    res = giatools.Image(
        data=arr,
        axes=giatools.default_normalized_axes,
        metadata=final_metadata,
    )

    # Squeeze singleton axes and save
    res = res.squeeze()
    print('Output TIFF shape:', res.data.shape)
    print('Output TIFF axes:', res.axes)
    print('Output TIFF', metadata_to_str(final_metadata))
    res.write(output_image_path, backend='tifffile')


def reduce_metadata(values: list[Any]) -> Any | None:
    non_none_values = list(filter(lambda value: value is not None, values))

    # Reduction is not possible if more than one type is involved (or none)
    value_types = [type(value) for value in non_none_values]
    if len(frozenset(value_types)) != 1:
        return None
    else:
        value_type = value_types[0]

    # For floating point types, reduce via arithmetic average
    if np.issubdtype(value_type, float):
        return np.mean(non_none_values)

    # For integer types, reduce via the median
    if np.issubdtype(value_type, int):
        return np.median(non_none_values)

    # For all other types, reduction is only possible if the values are identical
    if len(frozenset(non_none_values)) == 1:
        return non_none_values[0]
    else:
        return None


def metadata_to_str(metadata: dict) -> str:
    tokens = list()
    for key in sorted(metadata.keys()):
        value = metadata[key]
        if isinstance(value, tuple):
            value = '(' + ', '.join([f'{val}' for val in value]) + ')'
        tokens.append(f'{key}: {value}')
    return ', '.join(tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', type=str, nargs='+')
    parser.add_argument('out_file', type=str)
    parser.add_argument('axis', type=str)
    parser.add_argument('--preserve_values', default=False, action='store_true')
    args = parser.parse_args()

    concat_channels(
        args.input_files,
        args.out_file,
        args.axis,
        args.preserve_values,
    )
