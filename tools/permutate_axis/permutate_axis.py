import argparse
import sys
import warnings
import numpy as np
import skimage.io
import skimage.util 

def permutate_axis(input_image_path, output_image_path, axis, permutate):
    images = []
    raw_image = skimage.io.imread(input_image_path, plugin='tifffile')
    for i in permutate:
        # TODO generalise 
        if axis == 0:
            a_slice = raw_image[i]
        elif axis == 1:
            a_slice = raw_image[:,i]
        elif axis == 2:
            a_slice = raw_image[:,:,i]
        elif axis == 3:
            a_slice = raw_image[:,:,:,i]
        elif axis == 4:
            a_slice = raw_image[:,:,:,:,i]
        images.append(np.expand_dims(a_slice, axis))

    res = np.concatenate(images, axis)
    with warnings.catch_warnings():
    	warnings.simplefilter("ignore")
    	res = skimage.util.img_as_uint(res) #Attention: precision loss
    	skimage.io.imsave(output_image_path, res, plugin='tifffile')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=argparse.FileType('r'), help='input file')
    parser.add_argument('out_file', type=argparse.FileType('w'), help='out file (TIFF)')
    parser.add_argument('permutate', help='new channel order', default='0,1,2', type=str)
    parser.add_argument('--axis', dest='axis', type=int, default=0, help='concatenation axis')
    args = parser.parse_args()

    permutate = [int(item) for item in args.permutate.split(',')]
    permutate_axis(args.input_file.name, args.out_file.name, args.axis, permutate)
