import argparse

import numpy as np
import skimage.io
import skimage.util


def concat_channels(input_image_paths, output_image_path, axis, preserve_values):
    images = []
    for image_path in input_image_paths:

        raw_image = skimage.io.imread(image_path)
        if len(raw_image.shape) == 2:
            if axis == 0:
                raw_image = [raw_image]
            else:
                raw_image = np.expand_dims(raw_image, 2)

        # Preserve values: Convert to `float` dtype without changing the values
        if preserve_values:
            raw_image = raw_image.astype(float)

        # Preserve brightness: Scale values to 0..1
        else:
            raw_image = skimage.util.img_as_float(raw_image)

        images.append(raw_image)

    # Do the concatenation and save
    res = np.concatenate(images, axis)
    skimage.io.imsave(output_image_path, res, plugin='tifffile')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', type=argparse.FileType('r'), nargs='+')
    parser.add_argument('-o', dest='out_file', type=argparse.FileType('w'))
    parser.add_argument('--axis', dest='axis', type=int, default=0, choices=[0, 2])
    parser.add_argument('--preserve_values', default=False, action='store_true')
    args = parser.parse_args()

    concat_channels([x.name for x in args.input_files], args.out_file.name, args.axis, args.preserve_values)
