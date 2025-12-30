import argparse
import sys

import giatools.io
import pandas as pd
import skimage.io
import skimage.util


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter segmentation by features')
    parser.add_argument('input_file', type=argparse.FileType('r'), default=sys.stdin, help='input file')
    parser.add_argument('out_file', type=argparse.FileType('w'), default=sys.stdin, help='out file (TIFF)')
    parser.add_argument('feature_file', type=argparse.FileType('r'), default=sys.stdin, help='feature file (cols: label, f1, f2)')
    parser.add_argument('rule_file', type=argparse.FileType('r'), default=sys.stdin, help='file with rules per feature (cols: ,f1,2, rows: feature_name, min, max)')
    args = parser.parse_args()

    img_in = giatools.io.imreadraw(args.input_file.name)[0]
    features = pd.read_csv(args.feature_file, delimiter="\t")
    rules = pd.read_csv(args.rule_file, delimiter="\t")

    cols = [a for a in rules.columns if 'Unnamed' not in a]
    for a_c in cols:
        a_min = rules[rules.iloc[:, 0] == 'min'][a_c]
        a_max = rules[rules.iloc[:, 0] == 'max'][a_c]
        for a_l in features.label:
            a_val = float(features[features['label'] == a_l][a_c])
            if a_val < float(a_min) or a_val > float(a_max):
                img_in[img_in == int(a_l)] = 0

    res = skimage.util.img_as_uint(img_in)
    skimage.io.imsave(args.out_file.name, res, plugin="tifffile")
