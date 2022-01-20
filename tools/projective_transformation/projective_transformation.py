"""
Copyright 2019-2022 Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT

"""

import argparse
import warnings

import numpy as np
import pandas as pd
import skimage.color
import skimage.io
import tifffile
from scipy.ndimage import map_coordinates
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


def transform(moving_fn, fixed_fn, warp_mat, output_fn):

    moving = skimage.io.imread(moving_fn)
    nDims = len(moving.shape)
    assert nDims in [2, 3, 4, 5, 6], 'this tool only supports up to 6 dimensions'

    if moving.shape[-1] in [3, 4] and nDims > 2:
        isRGB = True
        moving = np.transpose(moving, (nDims - 1,) + tuple(_ for _ in range(nDims - 1)))
    else:
        isRGB = False

    if nDims > 3 or (nDims == 3 and not isRGB):
        isMulCh = True
    else:
        isMulCh = False

    fixed = skimage.io.imread(fixed_fn)
    if fixed.shape[-1] in [3, 4] and len(fixed.shape) > 2:
        hw_fixed = fixed.shape[-3:-1]
    else:
        hw_fixed = fixed.shape[-2:]

    if isRGB or isMulCh:
        shapeCh = moving.shape[0:-2]
        nCh = np.prod(shapeCh)
        moving = np.reshape(moving, (nCh,) + moving.shape[-2:])
        warped_moving = np.zeros((nCh,) + hw_fixed, dtype=moving.dtype)

    warp_mat = pd.read_csv(warp_mat, delimiter="\t", header=None)
    warp_mat = np.array(warp_mat)
    assert warp_mat.shape[0] in [3], 'only 2D image transformaton is supported'

    trans = ProjectiveTransform(matrix=warp_mat)
    warped_coords = warp_coords_batch(trans, hw_fixed)

    if isMulCh or isRGB:
        for i in range(nCh):
            warped_moving[i, ...] = map_coordinates(moving[i, ...], warped_coords, cval=0)
        warped_moving = np.reshape(warped_moving, shapeCh + warped_moving.shape[-2:])
        if isRGB:
            warped_moving = np.transpose(warped_moving, tuple(_ for _ in range(1, nDims)) + (0,))
    else:
        warped_moving = map_coordinates(moving, warped_coords, cval=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if isMulCh:
            tifffile.imwrite(output_fn, warped_moving, imagej=True, metadata={'mode': 'composite'})
        else:
            skimage.io.imsave(output_fn, warped_moving)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform the image")
    parser.add_argument("fixed_image", help="Path to the fixed image")
    parser.add_argument("moving_image", help="Path to the moving image (to be transformed)")
    parser.add_argument("warp_matrix", help="Path to the transformation matrix")
    parser.add_argument("warped_image", help="Path to the output (transfirmed moving image)")
    args = parser.parse_args()
    transform(args.moving_image, args.fixed_image, args.warp_matrix, args.warped_image)
