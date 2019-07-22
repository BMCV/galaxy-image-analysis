import skimage.io
import skimage.color
from skimage import img_as_uint
from skimage.exposure import equalize_adapthist
from PIL import Image
import numpy as np
import argparse
import sys


# TODO make importable by python script
def readImg(path):
    img = Image.open(open(str(path), 'rb'))

    if len(img.shape) > 2:
        img = skimage.color.rgb2gray(img)
    img = equalize_adapthist(img, clip_limit=0.03)
    img = img_as_uint(img)
    img = np.reshape(img, [img.shape[0], img.shape[1], 1])
    return img


parser = argparse.ArgumentParser()
parser.add_argument('input_file1', type=argparse.FileType('r'), default=sys.stdin, help='input file (red)')
parser.add_argument('input_file2', type=argparse.FileType('r'), default=sys.stdin, help='input file (green)')
parser.add_argument('out_file', type=argparse.FileType('w'), default=sys.stdin, help='out file (TIFF)')
args = parser.parse_args()

im1 = readImg(args.input_file1)
im2 = readImg(args.input_file2)
res = np.concatenate((im1, im2, np.zeros_like(im1)), axis=-1)
skimage.io.imsave(args.out_file, res)
