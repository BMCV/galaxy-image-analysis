#!/usr/bin/python

import argparse
import numpy as np
import os
import skimage.io
from skimage.measure import regionprops 

parser = argparse.ArgumentParser(description='Count Objects')
parser.add_argument('input_file', type=argparse.FileType('r'),
                    help='Label input file')
parser.add_argument('output_file', type=argparse.FileType('w'),
                    help='Tabular output file')
args = parser.parse_args() 

img_raw = skimage.io.imread(args.input_file.name)
res = len(regionprops(img_raw))

text_file = open(args.output_file.name, "w")
text_file.write("objects\n%s" % res)
text_file.close()
