import argparse

import giatools.io
import scipy.ndimage as ndi
import skimage.io
import skimage.util
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
    parser.add_argument('input', type=argparse.FileType('r'), help='Input file')
    parser.add_argument('output', type=argparse.FileType('w'), help='Output file (TIFF)')
    parser.add_argument('filter', choices=filters.keys(), help='Filter to be used')
    parser.add_argument('size', type=float, help='Size of the filter (e.g., radius, sigma)')
    args = parser.parse_args()

    im = giatools.io.imread(args.input.name)
    res = filters[args.filter](im, args.size)
    skimage.io.imsave(args.output.name, res, plugin='tifffile')
