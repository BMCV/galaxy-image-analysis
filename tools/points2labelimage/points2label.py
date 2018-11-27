import argparse
import sys
import numpy as np
import skimage.io
import pandas as pd

def points2label(labels, shape, output_file=None, is_TSV=False):
    labelimg = np.zeros([shape[0], shape[1]], dtype=np.int32)

    if is_TSV:
        df = pd.read_csv(labels, sep='\t')
    else:
        df = pd.read_csv(labels)

    for i in range(0, len(df)):
        a_row = df.iloc[i]
        print(df.iloc[i])
        labelimg[a_row[0], a_row[1]] = i+1

    if output_file is not None:
        skimage.io.imsave(output_file, labelimg)
    else:
        return labelimg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('label_file', type=argparse.FileType('r'), default=sys.stdin, help='label file')
    parser.add_argument('out_file', type=argparse.FileType('w'), default=sys.stdin, help='out file (TIFF)')
    parser.add_argument('org_file', type=argparse.FileType('r'), default=sys.stdin, help='input original file')
    parser.add_argument('--is_tsv', dest='is_tsv', type=bool, default=False, help='label file is TSV')
    args = parser.parse_args()

    original_shape = skimage.io.imread(args.org_file.name).shape

    points2label(args.label_file.name, original_shape, args.out_file.name, args.is_tsv)
