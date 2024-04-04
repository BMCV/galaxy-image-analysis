import argparse
import sys

import numpy as np
import skimage.io
import skimage.transform
import skimage.util
from PIL import Image


def scale_image(input_file, output_file, scale, order, antialias):
    Image.MAX_IMAGE_PIXELS = 50000 * 50000
    im = skimage.io.imread(input_file)

    # Parse `--scale` argument
    if ',' in scale:
        scale = [float(s.strip()) for s in scale.split(',')]
        assert len(scale) <= im.ndim, f'Image has {im.ndim} axes, but scale factors were given for {len(scale)} axes.'
        scale = scale + [1] * (im.ndim - len(scale))

    else:
        scale = float(scale)

        # For images with 3 or more axes, the last axis is assumed to correspond to channels
        if im.ndim >= 3:
            scale = [scale] * (im.ndim - 1) + [1]

    # Do the scaling
    res = skimage.transform.rescale(im, scale, order, anti_aliasing=antialias, preserve_range=True)

    # Preserve the `dtype` so that both brightness and range of values is preserved
    if res.dtype != im.dtype:
        if np.issubdtype(im.dtype, np.integer):
            res = res.round()
        res = res.astype(im.dtype)

    # Save result
    skimage.io.imsave(output_file, res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('out_file', type=argparse.FileType('w'), default=sys.stdin)
    parser.add_argument('--scale', type=str, required=True)
    parser.add_argument('--order', type=int, required=True)
    parser.add_argument('--antialias', default=False, action='store_true')
    args = parser.parse_args()

    scale_image(args.input_file.name, args.out_file.name, args.scale, args.order, args.antialias)
