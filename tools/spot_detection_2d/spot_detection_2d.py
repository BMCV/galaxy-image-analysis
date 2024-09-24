"""
Copyright 2021-2022 Biomedical Computer Vision Group, Heidelberg University.
Authors:
- Qi Gao (qi.gao@bioquant.uni-heidelberg.de)
- Leonid Kostrykin (leonid.kostrykin@bioquant.uni-heidelberg.de)

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
"""

import argparse

import giatools.io
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from numpy.typing import NDArray
from skimage.feature import blob_dog, blob_doh, blob_log

blob_filters = {
    'dog': blob_dog,
    'doh': blob_doh,
    'log': blob_log,
}


def mean_intensity(img: NDArray, y: int, x: int, radius: int) -> float:
    assert img.ndim == 2
    assert radius >= 0
    if radius == 0:
        return float(img[y, x])
    else:
        mask = np.ones(img.shape, bool)
        mask[y, x] = False
        mask = (ndi.distance_transform_edt(mask) <= radius)
        return img[mask].mean()


def spot_detection(
    fn_in: str,
    fn_out: str,
    frame_1st: int,
    frame_end: int,
    filter_type: str,
    min_scale: float,
    max_scale: float,
    abs_threshold: float,
    rel_threshold: float,
    boundary: int,
) -> None:

    # Load the single-channel 2-D input image (or stack thereof)
    stack = giatools.io.imread(fn_in)

    # Normalize input image so that it is a stack of images (possibly a stack of a single image)
    assert stack.ndim in (2, 3)
    if stack.ndim == 2:
        stack = stack.reshape(1, *stack.shape)

    # Slice the stack
    assert frame_1st >= 1
    assert frame_end >= 0
    stack = stack[frame_1st - 1:]
    if frame_end > 0:
        stack = stack[:-frame_end]

    # Select the blob detection filter
    assert filter_type.lower() in blob_filters.keys()
    blob_filter = blob_filters[filter_type.lower()]

    # Perform blob detection on each image of the stack
    detections = list()
    for img_idx, img in enumerate(stack):
        blobs = blob_filter(img, threshold=abs_threshold, threshold_rel=rel_threshold, min_sigma=min_scale, max_sigma=max_scale)
        for blob in blobs:
            y, x, scale = blob
            radius = scale * np.sqrt(2) * 2
            intensity = mean_intensity(img, round(y), round(x), round(radius))
            detections.append(
                {
                    'frame': img_idx + 1,
                    'pos_x': round(x),
                    'pos_y': round(y),
                    'scale': scale,
                    'radius': radius,
                    'intensity': intensity,
                }
            )

    # Build and save dataframe
    df = pd.DataFrame.from_dict(detections)
    df.to_csv(fn_out, index=False, float_format='%.2f', sep="\t")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Spot detection")

    parser.add_argument("fn_in", help="Name of input image or image sequence (stack).")
    parser.add_argument("fn_out", help="Name of output file to write the detections into.")
    parser.add_argument("frame_1st", type=int, help="Index for the starting frame to detect spots (1 for first frame of the stack).")
    parser.add_argument("frame_end", type=int, help="Index for the last frame to detect spots (0 for the last frame of the stack).")
    parser.add_argument("filter_type", help="Detection filter")
    parser.add_argument("min_scale", type=float, help="The minimum scale to consider for multi-scale detection.")
    parser.add_argument("max_scale", type=float, help="The maximum scale to consider for multi-scale detection.")
    parser.add_argument("abs_threshold", type=float, help=(
        "Filter responses below this threshold will be ignored. Only filter responses above this thresholding will be considered as blobs. "
        "This threshold is ignored if the relative threshold (below) corresponds to a higher response.")
    )
    parser.add_argument("rel_threshold", type=float, help=(
        "Same as the absolute threshold (above), but as a fraction of the overall maximal filter response of an image. "
        "This threshold is ignored if it corresponds to a response below the absolute threshold.")
    )
    parser.add_argument("boundary", type=int, help="Width of image boundaries (in pixel) where spots will be ignored.")

    args = parser.parse_args()
    spot_detection(args.fn_in, args.fn_out,
                   frame_1st=args.frame_1st, frame_end=args.frame_end,
                   filter_type=args.filter_type,
                   min_scale=args.min_scale, max_scale=args.max_scale,
                   abs_threshold=args.abs_threshold, rel_threshold=args.rel_threshold,
                   boundary=args.boundary)
