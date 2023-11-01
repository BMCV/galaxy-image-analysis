import argparse
import os
import warnings

import numpy as np
import pandas as pd
import skimage.io


def points2binaryimage(point_file, out_file, shape=[500, 500], has_header=False, invert_xy=False):

    img = np.zeros(shape, dtype=np.int16)
    if os.path.exists(point_file) and os.path.getsize(point_file) > 0:
        if has_header:
            df = pd.read_csv(point_file, skiprows=1, header=None, delimiter="\t")
        else:
            df = pd.read_csv(point_file, header=None, delimiter="\t")

        for i in range(0, len(df)):
            a_row = df.iloc[i]
            if int(a_row[0]) < 0 or int(a_row[1]) < 0:
                raise IndexError("Point {},{} is out of image with bounds {},{}.".format(int(a_row[0]), int(a_row[1]), shape[0], shape[1]))

            if invert_xy:
                if img.shape[0] <= int(a_row[0]) or img.shape[1] <= int(a_row[1]):
                    raise IndexError("Point {},{} is out of image with bounds {},{}.".format(int(a_row[0]), int(a_row[1]), shape[0], shape[1]))
                else:
                    img[int(a_row[1]), int(a_row[0])] = 32767
            else:
                if img.shape[0] <= int(a_row[1]) or img.shape[1] <= int(a_row[0]):
                    raise IndexError("Point {},{} is out of image with bounds {},{}.".format(int(a_row[1]), int(a_row[0]), shape[0], shape[1]))
                else:
                    img[int(a_row[0]), int(a_row[1])] = 32767
    else:
        raise Exception("{} is empty or does not exist.".format(point_file))  # appropriate built-in error?

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skimage.io.imsave(out_file, img, plugin='tifffile')  # otherwise we get problems with the .dat extension


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('point_file', type=argparse.FileType('r'), help='label file')
    parser.add_argument('out_file', type=str, help='out file (TIFF)')
    parser.add_argument('shapex', type=int, help='shapex')
    parser.add_argument('shapey', type=int, help='shapey')
    parser.add_argument('--has_header', dest='has_header', default=False, help='set True if CSV has header')
    parser.add_argument('--invert_xy', dest='invert_xy', default=False, help='invert x and y in CSV')

    args = parser.parse_args()

    # TOOL
    points2binaryimage(args.point_file.name, args.out_file, [args.shapey, args.shapex], has_header=args.has_header, invert_xy=args.invert_xy)
