import argparse
import sys
import warnings

import numpy as np
import pandas as pd
import skimage.io


def points2label(labels, shape, output_file=None, has_header=False, is_TSV=True):
    labelimg = np.zeros([shape[0], shape[1]], dtype=np.int32)

    if is_TSV:
        if has_header:
            df = pd.read_csv(labels, sep='\t', skiprows=1, header=None)
        else:
            df = pd.read_csv(labels, sep='\t', header=None)
    else:
        if has_header:
            df = pd.read_csv(labels, skiprows=1, header=None)
        else:
            df = pd.read_csv(labels, header=None)

    for i in range(0, len(df)):
        a_row = df.iloc[i]
        labelimg[a_row[0], a_row[1]] = i + 1

    if output_file is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skimage.io.imsave(output_file, labelimg, plugin='tifffile')
    else:
        return labelimg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('label_file', type=argparse.FileType('r'), default=sys.stdin, help='label file')
    parser.add_argument('out_file', type=argparse.FileType('w'), default=sys.stdin, help='out file')
    parser.add_argument('org_file', type=argparse.FileType('r'), default=sys.stdin, help='input original file')
    parser.add_argument('--has_header', dest='has_header', type=bool, default=False, help='label file has header')
    parser.add_argument('--is_tsv', dest='is_tsv', type=bool, default=True, help='label file is TSV')
    args = parser.parse_args()

    original_shape = skimage.io.imread(args.org_file.name, plugin='tifffile').shape

    points2label(args.label_file.name, original_shape, args.out_file.name, args.has_header, args.is_tsv)
