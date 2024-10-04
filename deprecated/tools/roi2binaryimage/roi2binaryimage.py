# Import packages
import argparse
import sys
import numpy as np
import skimage.io
import skimage.draw
import os
import ijroi
import scipy.ndimage
import shutil
from tifffile import imsave
from scipy import misc
 
# ROI2binaryimage transforms given .roi files into binary images containing the cells as
# described by the .roi files.
# Input 'input_folder' should be the folder in which you have your .roi files.
# Assumptions:
# - in 'input_folder', you have one or more folders, each containing a subfolder named 'RoiSet'
# that
# contains your .roi files.
# - in 'input_folder', you have additionally a single .tif file which is the original file related
#  to
# that manual segmentation represented by those .roi files.

# ROI2binaryimage creates an output folder in 'output_folder' containing two folders with the
# original
# images and the manual segmentations. The tif images carry the same names.

def ROI2binaryimage(input_folder,output_folder,size):

    compstruct = scipy.ndimage.generate_binary_structure(2, 2) # Mask for image dilation.
    normal_color = 4294901760 # Stroke color (chosen by me) for normal cells.
    stressed_color = 4278255360 # Stroke color (chosen by me) for stressed cells.

    # Create the result folders if they don't exist.
    # In the outputfolder as given to ROI2binaryimage, a folder is created named
    # 'Manual_segmentation_results'. In this folder, two subfolders are created;
    # 'Original_images' as well as 'Manual_segmentation'.
    if not(os.path.isdir(os.path.join(output_folder, "Manual_segmentation_results"))):
        os.makedirs(os.path.join(output_folder, "Manual_segmentation_results"))
        os.makedirs(os.path.join(output_folder, "Manual_segmentation_results",
                                 "Original_images"))
        os.makedirs(os.path.join(output_folder, "Manual_segmentation_results",
                                 "Manual_segmentations"))

    # Ignoring hidden folders such as .DS_store in Mac, find folders for images + rois.
    for folder in [p for p in os.listdir(input_folder) if not(p.startswith('.'))]:

        blank = np.zeros(size, dtype=np.int8)  # The image size
        canvas = [np.copy(blank), np.copy(blank)]  # Prep for background.
        overlap = [np.copy(blank), np.copy(blank)]  # Prep for the later overlapping images.

        # Find the original image and save it to the new location; also save its name.
        for image in [i for i in os.listdir(os.path.join(input_folder, folder)) if i
                .endswith('.tif')]:
            if not (np.all(size == list(np.asarray(misc.imread(os.path.join(input_folder,folder,image))).shape))):
                raise ValueError("Given/assumed size doesn't match image size")
            original_image_name = image
            shutil.copy2((os.path.join(input_folder, folder, image)),
                         (os.path.join(output_folder, "Manual_segmentation_results",
                                       "Original_images", image)))


        notif = original_image_name[0:-4]

        # Find all the .roi files in 'Roiset' and work with them.
        for file in [f for f in os.listdir(os.path.join(
                input_folder, folder, "Roiset")) if f.endswith('.roi')]:
            temp_canvas = [np.copy(blank),np.copy(blank)]

            with open((os.path.join(input_folder, folder, "Roiset", file)),"rb") as f:
                [roi,roi_color] = ijroi.read_roi(f)

            # Determine whether the points are valid (the roi has to fit in the given 'size')
            valid = np.logical_and(np.min(roi, axis=1) >= 0, roi[:,0] < size[0], roi[:,1] < size[1])
            valid_roi = roi[valid]
            x = valid_roi[:,0]
            y = valid_roi[:,1]


            # Differentiate between the normal and stressed cell rois.
            rr, cc = skimage.draw.polygon(x,y)
            if roi_color == normal_color:
                q = 0
            if roi_color == stressed_color:
                q = 1

            # Add roi to the image.
            canvas[q][rr,cc] = 1
            temp_canvas[q][rr,cc] = 1

            # Determine the overlap by adding the dilated version of the cell to the overlap file.
            # (This is to ensure the cells do not 'touch' as well as do not 'overlap').
            temp_canvas[q] = scipy.ndimage.binary_dilation(  # Dilate cell in all directions.
                temp_canvas[q], structure=compstruct).astype(temp_canvas[q].dtype)
            overlap[q] = overlap[q] + temp_canvas[q]


        # If there is an overlap of different cells, that area is set to 0.
        for ax in np.arange(size[0]):
            for ay in np.arange(size[1]):
                for i in range(len(overlap)):
                    if overlap[i][ax,ay] > 1:
                        canvas[i][ax,ay] = 0

        background = np.array([1 - np.clip((canvas[0]+canvas[1]), 0, 1)], ndmin=3, dtype=np.int8)

        image = np.array([canvas[0], canvas[1]])

        result = np.concatenate([background, image], axis=0)
        result = skimage.util.img_as_ubyte(result)

        imsave(os.path.join(output_folder, "Manual_segmentation_results",'Manual_segmentations',
            (notif + '.tif')), result)
    return None


# To run from command line.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', default=sys.stdin,
                        help='The folder with your input. Must contain subfolders, '
                             'each containing: 1. The original .tif image and 2. A folder named '
                             'RoiSet containing all the .roi files for that image. Must not '
                             'contain other folder than these. Files or folders starting with a . '
                             'are however ignored.')

    parser.add_argument('output_folder', nargs='?', default=os.getcwd(),
                        help='The folder in which you wish to create an output folder. '
                             'The output folder will end up containing: 1. a subfolder containing '
                             'all original .tif images and 2. a subfolder containing all binary '
                             'results, named after the original .tif images.')
    parser.add_argument('x', nargs='?',default=1398,
                        help='The size of your image in x-direction.')
    parser.add_argument('y', nargs='?', default=1948,
                        help='The size of your image in y-direction.')

    args = parser.parse_args()
    ROI2binaryimage(args.input_folder, args.output_folder,[args.y,args.x])
