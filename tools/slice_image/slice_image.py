import argparse
import os.path
import random
import warnings

import numpy as np
import skimage.feature
import skimage.io
import skimage.util


def slice_image(input_file, out_folder, window_size=64, stride=1, bg_thresh=1, limit_slices=False, n_thresh=5000, seed=None):

    # Primarily for testing purposes
    if seed is not None:
        random.seed(seed)

    img_raw = skimage.io.imread(input_file)
    if len(img_raw.shape) == 2:
        img_raw = np.expand_dims(img_raw, 3)

    with warnings.catch_warnings():  # ignore FutureWarning
        warnings.simplefilter("ignore")
        patches_raw = skimage.util.view_as_windows(img_raw, (window_size, window_size, img_raw.shape[2]), step=stride)
        patches_raw = patches_raw.reshape([-1, window_size, window_size, img_raw.shape[2]])

        new_path = os.path.join(out_folder, "%d.tiff")

        # samples for thresholding the amount of slices
        sample = random.sample(range(patches_raw.shape[0]), n_thresh)

        for i in range(0, patches_raw.shape[0]):
            # TODO improve
            sum_image = np.sum(patches_raw[i], 2) / img_raw.shape[2]

            if bg_thresh > 0:
                sum_image = skimage.util.img_as_uint(sum_image)
                g = skimage.feature.greycomatrix(sum_image, [1, 2], [0, np.pi / 2], nnormed=True, symmetric=True)
                hom = np.var(skimage.feature.greycoprops(g, prop='homogeneity'))
                if hom > bg_thresh:
                    continue

            if limit_slices:
                if i in sample:
                    res = skimage.util.img_as_uint(patches_raw[i])  # Attention: precision loss
                    skimage.io.imsave(new_path % i, res, plugin='tifffile')
            else:
                res = skimage.util.img_as_uint(patches_raw[i])  # Attention: precision loss
                skimage.io.imsave(new_path % i, res, plugin='tifffile')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=argparse.FileType('r'), help='Input file')
    parser.add_argument('out_folder', help='Output directory')
    parser.add_argument('--stride', dest='stride', type=int, default=1, help='Applied stride')
    parser.add_argument('--window_size', dest='window_size', type=int, default=64, help='Size of resulting patches')
    parser.add_argument('--bg_thresh', dest='bg_thresh', type=float, default=0, help='Skip background patches without information using a treshold')
    parser.add_argument('--n_thresh', dest='n_thresh', type=int, default=5000, help='Maximum number of slices to retain')
    parser.add_argument('--seed', dest='seed', type=int, default=None, help='Seed for random choice of slices')
    args = parser.parse_args()

    slice_image(
        args.input_file.name,
        args.out_folder,
        stride=args.stride,
        window_size=args.window_size,
        bg_thresh=args.bg_thresh,
        limit_slices=args.n_thresh > 0,
        n_thresh=args.n_thresh,
        seed=args.seed,
    )
