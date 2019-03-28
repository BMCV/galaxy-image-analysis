import argparse
import sys
import skimage.io
import skimage.transform
import scipy.misc
import warnings
import os
from PIL import Image


def scale_image(input_file, output_file, scale, order=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Image.MAX_IMAGE_PIXELS = 50000*50000
        img_in = skimage.io.imread(input_file)
        if order == 0:
            interp = 'nearest'
        elif order == 1:
            interp = 'bilinear'
        elif order == 2:
            interp = 'bicubic'

        if ',' in scale:
            scale = scale[1:-1].split(',')
            scale = [int(i) for i in scale]
        elif '.' in scale:
            scale = float(scale)
        else:
            scale = int(scale)

        res = scipy.misc.imresize(img_in, scale, interp=interp)
        skimage.io.imsave(output_file, res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=argparse.FileType('r'), default=sys.stdin, help='input file')
    parser.add_argument('out_file', type=argparse.FileType('w'), default=sys.stdin, help='out file (PNG)') 
    parser.add_argument('scale', type=str, help='fraction scaling factor(float), percentage scaling factor(int), output size(tuple(height,width))') # integer option not implemented in galaxy wrapper
    parser.add_argument('order', type=int, default=1, help='interpolation method')
    args = parser.parse_args()

    scale_image(args.input_file.name, args.out_file.name, args.scale, args.order)
