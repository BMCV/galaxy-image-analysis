import argparse
import sys
import warnings
import numpy as np
import skimage.io
import skimage.filters
import skimage.util
from skimage.morphology import disk

filterOptions = {
    'median' : lambda img_raw, radius: skimage.filters.median(img_raw, disk(radius)),
    'gaussian' : lambda img_raw, radius: skimage.filters.gaussian(img_raw, sigma=radius),
    'prewitt' : lambda img_raw, radius: skimage.filters.prewitt(img_raw),
    'sobel' : lambda img_raw, radius: skimage.filters.sobel(img_raw),
    'scharr' : lambda img_raw, radius: skimage.filters.scharr(img_raw),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=argparse.FileType('r'), default=sys.stdin, help='input file')
    parser.add_argument('out_file', type=argparse.FileType('w'), default=sys.stdin, help='out file (TIFF)')
    parser.add_argument('filter_type', choices=filterOptions.keys(), help='conversion type')
    parser.add_argument('radius', default=3.0, type=float, help='Radius/Sigma')
    args = parser.parse_args()

    img_in = skimage.io.imread(args.input_file.name)
    res = filterOptions[args.filter_type](img_in, args.radius)
    res[res<-1]=-1
    res[res>1]=1

    with warnings.catch_warnings():
    	warnings.simplefilter("ignore")
    	res = skimage.util.img_as_uint(res) #Attention: precision loss
    	skimage.io.imsave(args.out_file.name, res, plugin='tifffile')
