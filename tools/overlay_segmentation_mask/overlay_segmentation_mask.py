import argparse
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#TODO make importable by python script

import skimage.io
import skimage.measure

parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=argparse.FileType('r'), help='input file')
parser.add_argument('mask_file', type=argparse.FileType('r'), help='mask file')
parser.add_argument('out_file', type=str, help='out file (PNG)') # file would be created immediately with argparse.FileType('w') s.t. file cannot be renamed on galaxy
parser.add_argument('--grey', dest='greyscale', action='store_true', help='image is greyscale')
parser.add_argument('--label', dest='label', action='store_true', help='plot label')
parser.add_argument('--label_color', dest='label_color', default='#FFFF00', help='label color')
parser.add_argument('--thickness', dest='thickness', default=0.3, type=float, help='thickness')
parser.add_argument('--stroke_color', dest='stroke_color', default='#ff0000', help='stroke color')
args = parser.parse_args()
img = skimage.io.imread(args.input_file.name)
label = skimage.io.imread(args.mask_file.name)

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')

if args.label:
    for reg in skimage.measure.regionprops(label):
        ax.text(reg.centroid[1], reg.centroid[0], str(reg.label), color=args.label_color)

if args.greyscale:
    plt.imshow(img, cmap=plt.cm.gray)
else:
    plt.imshow(img)
plt.contour(label, linewidths=args.thickness, colors=args.stroke_color)

fig.canvas.print_png(args.out_file)