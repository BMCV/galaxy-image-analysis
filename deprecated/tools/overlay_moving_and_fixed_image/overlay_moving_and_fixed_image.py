import argparse

import numpy as np
import pandas as pd
import skimage.color
import skimage.io
from PIL import Image
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


def transform(moving_image, fixed_image, warp_matrix):
    trans = ProjectiveTransform(matrix=warp_matrix)
    warped_coords = warp_coords_batch(trans, fixed_image.shape)
    return map_coordinates(moving_image, warped_coords)


def overlay(moving_image, fixed_image, factor, overlay_out_path):
    moving_image = Image.fromarray(moving_image).convert("RGBA")
    fixed_image = Image.fromarray(fixed_image).convert("RGBA")
    overlay_out = Image.blend(moving_image, fixed_image, factor)
    overlay_out.save(overlay_out_path, "PNG")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay two images")
    parser.add_argument("fixed_image", help="Path to fixed image")
    parser.add_argument("moving_image", help="Path to moving image")
    parser.add_argument("warp_matrix", help="Paste path to warp_matrix.csv that should be used for transformation")
    parser.add_argument("--inverse_transform", dest='inverse_transform', action='store_true', help="Set if inverse transform should be visualized")
    parser.add_argument("--factor", dest="factor", help="Enter the factor by which images should be blended, 1.0 returns a copy of second image", type=float, default=0.5)
    parser.add_argument("overlay_out", help="Overlay output path")
    args = parser.parse_args()

    fixed_image = skimage.io.imread(args.fixed_image)
    moving_image = skimage.io.imread(args.moving_image)

    warp_matrix = pd.read_csv(args.warp_matrix, delimiter="\t", header=None)
    warp_matrix = np.array(warp_matrix)
    if args.inverse_transform:
        fixed_image = transform(fixed_image, moving_image, warp_matrix)
    else:
        warp_matrix = np.linalg.inv(warp_matrix)
        moving_image = transform(moving_image, fixed_image, warp_matrix)

    overlay(moving_image, fixed_image, args.factor, args.overlay_out)
