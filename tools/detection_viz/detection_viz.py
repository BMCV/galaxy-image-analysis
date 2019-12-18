import argparse
import sys
import os
import csv
 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import skimage.io

def plot_circles(file_name, ax, color, stroke_size, radius):
    resfile = open(file_name, 'rb')
    rd = csv.reader(resfile, delimiter=',')
    for row in rd:
        circ = plt.Circle((int(row[1]), int(row[0])), lw=stroke_size, radius=radius, color=color, fill=False)
        ax.add_patch(circ)
    resfile.close()

def detection_viz(input_file, output_file, tp=None, fn=None, fp=None, stroke_size=3, circle_radius=50):
    img = skimage.io.imread(input_file)

    fig = plt.figure(figsize=(40, 40))
    ax = fig.add_axes([0, 0, 1, 1]) 
    ax.axis('off')

    plt.imshow(img)
    if tp is not None:
        plot_circles(tp, ax, '#00FF00', stroke_size, circle_radius)
    if fn is not None:
        plot_circles(fn, ax, 'red', stroke_size, circle_radius)
    if fp is not None:
        plot_circles(fp, ax, 'darkorange', stroke_size, circle_radius)

    fig.canvas.print_png(output_file, dpi=1800)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=argparse.FileType('r'), help='original file')
    # output file should not be of type argparse.FileType('w') sine it is created immediately in this case which leads to an error in renaming
    parser.add_argument('out_file_str', type=str, help='string of output file name') 
    parser.add_argument('--tp', dest='input_tp_file', type=argparse.FileType('r'), help='input TP file')
    parser.add_argument('--fn', dest='input_fn_file', type=argparse.FileType('r'), help='input FN file')
    parser.add_argument('--fp', dest='input_fp_file', type=argparse.FileType('r'), help='input FP file')
    parser.add_argument('--stroke_size', dest='thickness', default=3, type=float, help='stroke thickness')
    parser.add_argument('--circle_radius', dest='circle_radius', type=float, default=50, help='circle radius')
    args = parser.parse_args()

    tp=None
    if args.input_tp_file:
        tp=args.input_tp_file.name
    fn=None
    if args.input_fn_file:
        fn=args.input_fn_file.name
    fp=None
    if args.input_fp_file:
        fp=args.input_fp_file.name

    detection_viz(args.input_file.name, args.out_file_str, tp=tp, fn=fn, fp=fp, stroke_size=args.thickness, circle_radius=args.circle_radius)
