import argparse
import csv

import numpy as np
import orientationpy
import skimage.io
import skimage.util


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--sigma', type=float, required=True)
    parser.add_argument('--min_coherency', type=float, required=True)
    parser.add_argument('--min_energy', type=float, required=True)
    parser.add_argument('--max_precision', type=int, required=True)
    parser.add_argument('--output_angle_tsv', type=str, default=None)
    args = parser.parse_args()

    im = skimage.io.imread(args.input)
    im = skimage.util.img_as_float(im)
    im = np.squeeze(im)
    assert im.ndim == 2

    Gy, Gx = orientationpy.computeGradient(im, mode=args.mode)
    structureTensor = orientationpy.computeStructureTensor([Gy, Gx], sigma=args.sigma)
    orientations = orientationpy.computeOrientation(structureTensor, computeEnergy=True, computeCoherency=True)

    # Compute angle according to https://bigwww.epfl.ch/demo/orientationj/#dist:
    mask = np.logical_and(
        orientations['coherency'] >= args.min_coherency,
        orientations['energy'] >= args.min_energy * orientations['energy'].max(),
    )
    angles = orientations['theta'][mask]
    weights = orientations['coherency'][mask]
    bin_size = 1 if args.max_precision == 0 else pow(10, -args.max_precision)
    hist, bin_edges = np.histogram(
        angles,
        range=(-90, +90),
        weights=weights,
        bins=round(180 / bin_size),
    )
    hidx = np.argmax(hist)
    angle = (bin_edges[hidx] + bin_edges[hidx + 1]) / 2
    angle = round(angle, args.max_precision)

    # Write results
    if args.output_angle_tsv:
        with open(args.output_angle_tsv, 'w') as fp:
            writer = csv.writer(fp, delimiter='\t', lineterminator='\n')
            writer.writerow(['Angle'])
            writer.writerow([angle])
