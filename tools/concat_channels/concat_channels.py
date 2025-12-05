import argparse

import giatools
import numpy as np
import skimage.io
import skimage.util


normalized_axes = 'QTZYXC'


def concat_channels(
    input_image_paths: list[str],
    output_image_path: str,
    axis: str,
    preserve_values: bool,
):
    # Create list of arrays to be concatenated
    images = []
    for image_path in input_image_paths:

        img = giatools.Image.read(image_path, normalize_axes=normalized_axes)
        arr = img.data

        # Preserve values: Convert to `float` dtype without changing the values
        if preserve_values:
            arr = arr.astype(float)

        # Preserve brightness: Scale values to 0..1
        else:
            arr = skimage.util.img_as_float(arr)

        images.append(arr)

    # Do the concatenation
    axis_pos = normalized_axes.index(axis)
    arr = np.concatenate(images, axis_pos)
    res = giatools.Image(arr, normalized_axes)

    # Squeeze singleton axes and save
    squeezed_axes = ''.join(np.array(list(res.axes))[np.array(arr.shape) > 1])
    res = res.squeeze_like(squeezed_axes)
    res.write(output_image_path, backend='tifffile')


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
