import skimage.io
from skimage.transform import ProjectiveTransform
from scipy.ndimage import map_coordinates
import numpy as np
import pandas as pd
import argparse
import warnings
import shutil 


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

    for i in range(0, (tf_coords.shape[0]//batch_size+1)):
        tf_coords[batch_size*i:batch_size*(i+1)] = coord_map(tf_coords[batch_size*i:batch_size*(i+1)])
    tf_coords = tf_coords.T.reshape((-1, cols, rows)).swapaxes(1, 2)

    _stackcopy(coords[1, ...], tf_coords[0, ...])
    _stackcopy(coords[0, ...], tf_coords[1, ...])
    if len(shape) == 3:
        coords[2, ...] = range(shape[2])

    return coords


def transform(moving_image, fixed_image, warp_matrix, out):
    moving_image = skimage.io.imread(moving_image)
    fixed_image = skimage.io.imread(fixed_image)
    warp_matrix = pd.read_csv(warp_matrix, delimiter="\t", header=None)
    warp_matrix = np.array(warp_matrix)

    trans = ProjectiveTransform(matrix=warp_matrix)
    warped_coords = warp_coords_batch(trans, fixed_image.shape)
    t = map_coordinates(moving_image, warped_coords, mode='reflect')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skimage.io.imsave(out, t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform the image")
    parser.add_argument("fixed_image", help="Paste path to image.png that should be transformed")
    parser.add_argument("moving_image", help="Paste path to fixed image.png")
    parser.add_argument("warp_matrix", help="Paste path to warp_matrix.csv that should be used for transformation")
    parser.add_argument("out", help="Paste path to file in which transformed image should be saved")
    args = parser.parse_args()
    transform(args.moving_image, args.fixed_image, args.warp_matrix, args.out)
