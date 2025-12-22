import argparse
import json

import giatools
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input image filepath')
    parser.add_argument('output', type=str, help='Output image filepath (TIFF)')
    parser.add_argument('params', type=str)
    args = parser.parse_args()

    # Read the config file
    with open(args.params) as cfgf:
        cfg = json.load(cfgf)


    # Read the input image
    image = giatools.Image.read(args.input)
    print('Input image shape:', image.data.shape)
    print('Input image axes:', image.axes)
    print('Input image dtype:', image.data.dtype)

    # Perform the clipping
    clip_args = [
        cfg.get('lower_bound', -np.inf),
        cfg.get('upper_bound', +np.inf),
    ]
    print('Applying clipping:', str(clip_args))
    image.data = image.data.clip(*clip_args).astype(image.data.dtype)

    # Write the result
    image = image.normalize_axes_like(
        image.original_axes,
    )
    print('Output image shape:', image.data.shape)
    print('Output image axes:', image.axes)
    print('Output image dtype:', image.data.dtype)
    image.write(args.output, backend='tifffile')
