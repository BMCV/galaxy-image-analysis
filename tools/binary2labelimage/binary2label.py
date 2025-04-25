import argparse

import giatools
import scipy.ndimage as ndi
import tifffile


# Parse CLI parameters
parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='input file')
parser.add_argument('output', type=str, help='output file (TIFF)')
args = parser.parse_args()

# Read the input image
img = giatools.Image.read(args.input)

# Make sure the image is truly binary
img_arr_bin = (img.data > 0)

# Perform the labeling
img.data = ndi.label(img_arr_bin)[0]

# Write the result image (same axes as input image)
tifffile.imwrite(args.output, img.data, metadata=dict(axes=img.axes))
