import argparse
import sys
import warnings
import numpy as np
import skimage.io
import skimage.util

def concat_channels(input_image_paths, output_image_path, axis):
    images = []
    for image_path in input_image_paths:
    	raw_image = skimage.io.imread(image_path)
        if len(raw_image.shape) == 2:
            if axis == 0:
                raw_image = [raw_image]
            else:
                raw_image = np.expand_dims(raw_image, 2)
        images.append(raw_image)
    res = np.concatenate(images, axis)
    with warnings.catch_warnings():
    	warnings.simplefilter("ignore")
    	res = skimage.util.img_as_uint(res) #Attention: precision loss
    	skimage.io.imsave(output_image_path, res, plugin='tifffile')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', type=argparse.FileType('r'), nargs='+', help='input file')
    parser.add_argument('-o', dest='out_file', type=argparse.FileType('w'), help='out file (TIFF)')
    parser.add_argument('--axis', dest='axis', type=int, default=0, choices=[0,2], help='concatenation axis')
    args = parser.parse_args()

    concat_channels(args.input_files, args.out_file.name, args.axis)
