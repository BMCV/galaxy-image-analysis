"""
Copyright 2017-2022 Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT

"""

import argparse

import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from skimage.measure import ransac
from skimage.transform import AffineTransform


def landmark_registration(pts_f1, pts_f2, out_f, res_th=None, max_ite=None, delimiter="\t"):

    points1 = pd.read_csv(pts_f1, delimiter=delimiter)
    points2 = pd.read_csv(pts_f2, delimiter=delimiter)

    src = np.concatenate([np.array(points1['x']).reshape([-1, 1]),
                          np.array(points1['y']).reshape([-1, 1])],
                         axis=-1)
    dst = np.concatenate([np.array(points2['x']).reshape([-1, 1]),
                          np.array(points2['y']).reshape([-1, 1])],
                         axis=-1)

    if res_th is not None and max_ite is not None:
        model_robust, inliers = ransac((src, dst), AffineTransform, min_samples=3, residual_threshold=res_th, max_trials=max_ite)
        pd.DataFrame(model_robust.params).to_csv(out_f, header=None, index=False, sep="\t")

    else:
        A = np.zeros((src.size, 6))
        A[0:src.shape[0], [0, 1]] = src
        A[0:src.shape[0], 2] = 1
        A[src.shape[0]:, [3, 4]] = src
        A[src.shape[0]:, 5] = 1
        b = dst.T.flatten().astype('float64')
        x = lstsq(A, b)

        tmat = np.eye(3)
        tmat[0, :] = x[0].take([0, 1, 2])
        tmat[1, :] = x[0].take([3, 4, 5])
        pd.DataFrame(tmat).to_csv(out_f, header=None, index=False, sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate affine transformation matrix based on landmark coordinates")
    parser.add_argument("fn_pts1", help="Coordinates of SRC landmarks (tsv file)")
    parser.add_argument("fn_pts2", help="Coordinates of DST landmarks (tsv file)")
    parser.add_argument("fn_tmat", help="Path the output (transformation matrix)")
    parser.add_argument("--res_th", dest="res_th", type=float, help="Maximum distance for a data point to be classified as an inlier")
    parser.add_argument("--max_ite", dest="max_ite", type=int, help="Maximum number of iterations for random sample selection")
    args = parser.parse_args()

    res_th = None
    if args.res_th:
        res_th = args.res_th
    max_ite = None
    if args.max_ite:
        max_ite = args.max_ite

    landmark_registration(args.fn_pts1, args.fn_pts2, args.fn_tmat, res_th=res_th, max_ite=max_ite)
