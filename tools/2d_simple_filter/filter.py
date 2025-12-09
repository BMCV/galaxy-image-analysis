import argparse

import giatools
import scipy.ndimage as ndi
from skimage.morphology import disk


filters = {
    'gaussian': lambda im, sigma: ndi.gaussian_filter(im, sigma),
    'median': lambda im, radius: ndi.median_filter(im, footprint=disk(radius)),
    'prewitt_h': lambda im, *args: ndi.prewitt(im, axis=1),
    'prewitt_v': lambda im, *args: ndi.prewitt(im, axis=0),
    'sobel_h': lambda im, *args: ndi.sobel(im, axis=1),
    'sobel_v': lambda im, *args: ndi.sobel(im, axis=0),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input image filepath')
    parser.add_argument('output', type=str, help='Output image filepath (TIFF)')
    parser.add_argument('filter', choices=filters.keys(), help='Filter to be used')
    parser.add_argument('size', type=float, help='Size of the filter (e.g., radius, sigma)')
    args = parser.parse_args()

    # Read the input image, perform filtering, write output (preserve metadata)
    img = giatools.Image.read(args.input)
    img.data = filters[args.filter](img.data, args.size)
    img.normalize_axes_like(img.original_axes).write(args.output, backend='tifffile')
