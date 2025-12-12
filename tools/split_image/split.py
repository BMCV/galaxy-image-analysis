import argparse
import math

import giatools
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('axis', type=str)
parser.add_argument('output', type=str)
parser.add_argument('--squeeze', action='store_true', default=False)
args = parser.parse_args()

# Validate and normalize input parameters
assert len(args.axis) == 1, 'Axis input must be a single character.'
axis = args.axis.replace('S', 'C')

# Read input image with normalized axes
img_in = giatools.Image.read(args.input)

# Determine the axis to split along
axis_pos = img_in.axes.index(axis)

# Perform the splitting
arr = np.moveaxis(img_in.data, axis_pos, 0)
decimals = math.ceil(math.log10(1 + arr.shape[0]))
output_filename_pattern = f'{args.output}/%0{decimals}d.tiff'
for img_idx, img in enumerate(arr):
    img = np.moveaxis(img[None], 0, axis_pos)

    # Construct the output image, remove axes added by normalization
    img_out = giatools.Image(
        data=img,
        axes=img_in.axes,
        metadata=img_in.metadata,
    ).squeeze_like(
        img_in.original_axes,
    )

    # Optionally, squeeze the image
    if args.squeeze:
        img_out = img_out.squeeze()

    # Save the result
    img_out.write(
        output_filename_pattern % (img_idx + 1),
    )
