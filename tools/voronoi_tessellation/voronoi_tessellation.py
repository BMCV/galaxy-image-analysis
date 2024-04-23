import argparse

import giatools.io
import numpy as np
import scipy.ndimage as ndi
import skimage.io
from skimage.segmentation import watershed


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    im = giatools.io.imread(args.input)
    im = im.squeeze()
    assert im.ndim == 2

    distances = np.full(im.shape, np.inf)
    for label in np.unique(im):
        if label == 0:
            continue

        label_distances = ndi.distance_transform_edt(im != label)
        distances = np.min((distances, label_distances), axis=0)

    result = watershed(
        image=distances,
        markers=im,
    )

    skimage.io.imsave(args.output, result)
