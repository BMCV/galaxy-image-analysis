from imageio import imread as io_imread
from skimage.measure import regionprops
import numpy as np
#import matplotlib.pyplot as plt
import scipy
import skimage.io
import skimage.draw
from tifffile import imsave
import os
import argparse
import warnings

# split_label_image takes a label image and outputs a similar file with the given name where the labeled
# parts of the image that touch (or overlap) are separated by at least 1 pixel (at most 2).


def split_labelmap(labelmap,outputfile):

    # Information from the label map.
    label_img = io_imread(labelmap)
    xtot, ytot = label_img.shape
    props = regionprops(label_img)
    N = len(props)

    # Creating the backgrounds.
    background = np.zeros([xtot,ytot], 'uint8')
    overlap = np.zeros([N,xtot,ytot],'uint8')
    compstruct = scipy.ndimage.generate_binary_structure(2, 2)  # Mask for image dilation.

    i = 0
    for cell in props:
        cell_image = cell.image.astype('uint8')
        #plt.imshow(cell_image)

        # Replace the background area corresponding to the bounding box with the image representing the cell.
        background[int(cell.bbox[0]):int(cell.bbox[2]),int(cell.bbox[1]):int(cell.bbox[3])] += cell_image
        overlap[i][int(cell.bbox[0]):int(cell.bbox[2]), int(cell.bbox[1]):int(cell.bbox[3])] = cell_image

        # In the overlap array, dilate the cell in all directions.
        overlap[i] = scipy.ndimage.binary_dilation(
            overlap[i], structure=compstruct).astype(overlap[i].dtype)

        i += 1

    if len(props) > 1:
        # Sum together the overlap.
        total_overlap = sum(overlap)
    
        # Wherever the overlap is greater than 1 replace that point with zero in the final image.
        for x in range(xtot):
            for y in range(ytot):
                if total_overlap[x,y] > 1:
                    background[x,y] = 0

    # Force the image into 8-bit.
    result = skimage.util.img_as_ubyte(background)

    # Save image
    with warnings.catch_warnings(): 
        warnings.simplefilter("ignore")
        skimage.io.imsave(outputfile, result, plugin="tifffile")

    return None

# To run from command line.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('labelmap',
                        help='Label map image.')
    parser.add_argument('outputfile',
                        help='Output file. Without extension (although it corrects if you '
                             'add it; will always return a .tif')

    args = parser.parse_args()
    split_labelmap(args.labelmap, args.outputfile)
