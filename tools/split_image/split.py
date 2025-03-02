import argparse

import giatools
import numpy as np
import tifffile


def str_without_positions(s: str, positions: list[int]) -> str:
    """
    Returns the string `s` with the `characters` removed from it.
    """
    for pos in sorted(positions, reverse=True):
        s = s[:pos] + s[pos + 1:]
    return s


parser = argparse.Parser()
parser.add_argument('input', type=str)
parser.add_argument('axis', type=str)
parser.add_argument('output', type=str)
parser.add_argument('--squeeze', type=bool, action='store_true', default=False)
args = parser.parse_args()

# Validate and normalize input parameters
assert len(args.axis) == 1
axis = args.axis.replace('S', 'C')

# Read input image as TZYXC
img_in = giatools.imread(args.input)
axes = 'TZYXC'

# Determine the axis to split along
axis_pos = axes.index(axis)

# Perform the splitting
arr = np.moveaxis(img_in, axis_pos, 0)
for img_idx, img in enumerate(arr):
    img = np.moveaxis(img[None], 0, axis_pos)

    # Optionally, squeeze the image
    s = [axis_pos for axis_pos in range(len(axes)) if img.shape[axis_pos] == 1 and axes[axis_pos] not in 'YX']
    img = np.squeeze(img, axis=s)
    img_axes = str_without_positions(axes, s)

    # Save the result
    tifffile.imwrite('', img, metadata=dict(axes=img_axes))
