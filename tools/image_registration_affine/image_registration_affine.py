"""
Copyright 2021-2022 Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT

"""
import argparse

import numpy as np
import pandas as pd
import skimage.io
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.signal import convolve2d
from skimage.color import rgb2gray, rgba2rgb
from skimage.filters import gaussian
from skimage.transform import ProjectiveTransform


def _stackcopy(a, b):
    if a.ndim == 3:
        a[:] = b[:, :, np.newaxis]
    else:
        a[:] = b


def warp_coords_batch(coord_map, shape, dtype=np.float64, batch_size=1000000):
    rows, cols = shape[0], shape[1]
    coords_shape = [len(shape), rows, cols]
    if len(shape) == 3:
        coords_shape.append(shape[2])
    coords = np.empty(coords_shape, dtype=dtype)

    tf_coords = np.indices((cols, rows), dtype=dtype).reshape(2, -1).T

    for i in range(0, (tf_coords.shape[0] // batch_size + 1)):
        tf_coords[batch_size * i:batch_size * (i + 1)] = coord_map(tf_coords[batch_size * i:batch_size * (i + 1)])
    tf_coords = tf_coords.T.reshape((-1, cols, rows)).swapaxes(1, 2)

    _stackcopy(coords[1, ...], tf_coords[0, ...])
    _stackcopy(coords[0, ...], tf_coords[1, ...])
    if len(shape) == 3:
        coords[2, ...] = range(shape[2])

    return coords


def affine_registration(params, moving, fixed, metric='mae'):
    tmat = np.eye(3)
    tmat[0, :] = params.take([0, 1, 2])
    tmat[1, :] = params.take([3, 4, 5])

    trans = ProjectiveTransform(matrix=tmat)
    warped_coords = warp_coords_batch(trans, fixed.shape)
    t = map_coordinates(moving, warped_coords, mode='nearest')
    f = fixed

    if metric == 'mse':
        err = (t - f) ** 2

    elif metric == 'mae':
        err = (t - f)

    elif metric == 'lcc':
        sum_filt = np.ones((9, 9))
        win_size = 81

        t_sum = convolve2d(t, sum_filt, mode='same', boundary='symm')
        f_sum = convolve2d(f, sum_filt, mode='same', boundary='symm')
        t2_sum = convolve2d(t * t, sum_filt, mode='same', boundary='symm')
        f2_sum = convolve2d(f * f, sum_filt, mode='same', boundary='symm')
        tf_sum = convolve2d(t * f, sum_filt, mode='same', boundary='symm')

        cross = tf_sum - f_sum * t_sum / win_size
        t_var = t2_sum - t_sum * t_sum / win_size
        f_var = f2_sum - f_sum * f_sum / win_size
        cc = cross * cross / (t_var * f_var + 1e-5)
        err = 1 - cc

    return err.flatten()


def read_img_as_gray(fn):
    im = skimage.io.imread(fn)
    nDims = len(im.shape)
    assert nDims in [2, 3], 'this tool does not support multichannel images'
    if nDims == 3:
        assert im.shape[-1] in [3, 4], 'this tool does not support multichannel images'
        if im.shape[-1] == 4:
            im = rgba2rgb(im)
        im = rgb2gray(im)
    im = im.astype(float)
    im = im / np.max(im)
    return im


def image_registration(fn_moving, fn_fixed, fn_out, smooth_sigma=3, metric='lcc'):
    moving = read_img_as_gray(fn_moving)
    fixed = read_img_as_gray(fn_fixed)

    moving = gaussian(moving, sigma=smooth_sigma)
    fixed = gaussian(fixed, sigma=smooth_sigma)

    x = np.array([1, 0, 0, 0, 1, 0], dtype='float64')
    result = least_squares(affine_registration, x, args=(moving, fixed, metric))

    tmat = np.eye(3)
    tmat[0, :] = result.x.take([0, 1, 2])
    tmat[1, :] = result.x.take([3, 4, 5])

    pd.DataFrame(tmat).to_csv(fn_out, header=None, index=False, sep="\t")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Estimate the transformation matrix")
    parser.add_argument("fn_moving", help="Path to the moving image")
    parser.add_argument("fn_fixed", help="Path to the fixed (reference) image")
    parser.add_argument("fn_tmat", help="Path to the output file for saving the transformation matrix")
    parser.add_argument("sigma", type=float, help="Sigma of Gaussian filter for smoothing input images")
    parser.add_argument("metric", help="Image similarity metric")
    args = parser.parse_args()

    image_registration(args.fn_moving, args.fn_fixed, args.fn_tmat, args.sigma, args.metric)
