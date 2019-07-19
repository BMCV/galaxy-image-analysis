import argparse
import openslide
import os
import fnmatch
import skimage.io
import numpy as np
 
def wsi_extract_top_view(input_path, out_path):
    img_raw = openslide.OpenSlide(input_path)
    top_size = img_raw.level_dimensions[len(img_raw.level_dimensions)-1]
    img_area = img_raw.read_region((0,0), len(img_raw.level_dimensions)-1, top_size)
    img_area = np.asarray(img_area, dtype=np.uint8)
    skimage.io.imsave(out_path, img_area, plugin="tifffile")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=argparse.FileType('r'), help='input file')
    parser.add_argument('out_file', help='out file')
    args = parser.parse_args()

    wsi_extract_top_view(args.input_file.name, args.out_file)
