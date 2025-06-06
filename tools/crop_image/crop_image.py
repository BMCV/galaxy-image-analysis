import argparse
import os

import numpy as np
from giatools.image import Image


def crop_image(
    image_filepath: str,
    labelmap_filepath: str,
    output_ext: str,
    output_dir: str,
    skip_labels: frozenset[int],
):
    image = Image.read(image_filepath)
    labelmap = Image.read(labelmap_filepath)

    if image.axes != labelmap.axes:
        raise ValueError(f'Axes mismatch between image ({image.axes}) and label map ({labelmap.axes}).')

    if image.data.shape != labelmap.data.shape:
        raise ValueError(f'Shape mismatch between image ({image.data.shape}) and label map ({labelmap.data.shape}).')

    for label in np.unique(labelmap.data):
        if label in skip_labels:
            continue
        roi_mask = (labelmap.data == label)
        roi = crop_image_to_mask(image.data, roi_mask)
        roi_image = Image(roi, image.axes).normalize_axes_like(image.original_axes)
        roi_image.write(os.path.join(output_dir, f'{label}.{output_ext}'))


def crop_image_to_mask(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Crop the `data` array to the minimal bounding box in `mask`.

    The arguments are not modified.
    """
    assert data.shape == mask.shape

    # Crop `data` to the convex hull of the mask in each dimension
    for dim in range(data.ndim):
        mask1d = mask.any(axis=tuple((i for i in range(mask.ndim) if i != dim)))
        mask1d_indices = np.where(mask1d)[0]
        mask1d_indices_cvxhull = np.arange(min(mask1d_indices), max(mask1d_indices) + 1)
        data = data.take(axis=dim, indices=mask1d_indices_cvxhull)

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str)
    parser.add_argument('labelmap', type=str)
    parser.add_argument('skip_labels', type=str)
    parser.add_argument('output_ext', type=str)
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()

    crop_image(
        image_filepath=args.image,
        labelmap_filepath=args.labelmap,
        output_ext=args.output_ext,
        output_dir=args.output_dir,
        skip_labels=frozenset((int(label.strip()) for label in args.skip_labels.split(','))),
    )
