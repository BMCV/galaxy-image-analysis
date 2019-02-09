import skimage.io
import skimage.color
import numpy as np
import os
import sys
import warnings

#TODO make importable by python script

args = sys.argv

def readImg(path):
    img = skimage.io.imread(path)
    if len(img.shape) > 2:
        img = skimage.color.rgb2gray(img)
    img = np.expand_dims(img > 0, 3)
    return img

im1 = readImg(args[1])
im2 = readImg(args[2])
res = np.concatenate((im1, im2, np.zeros_like(im1)), axis=2) * 1.0

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    skimage.io.imsave(args[3], res)
