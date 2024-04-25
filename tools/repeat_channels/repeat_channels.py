import argparse

import giatools.io
import numpy as np
import skimage.io


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('count', type=int)
    parser.add_argument('output', type=str)
    args = parser.parse_args()

    im = giatools.io.imread(args.input)
    im = np.squeeze(im)
    im = np.dstack([im] * args.count)
    skimage.io.imsave(args.output, im)
