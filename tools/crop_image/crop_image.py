import argparse
import os

import dask.array as da
import giatools
import giatools.image
import numpy as np

# Fail early if an optional backend is not available
giatools.require_backend('omezarr')


def crop_image(
    image_filepath: str,
    labelmap_filepath: str,
    output_ext: str,
    output_dir: str,
    skip_labels: frozenset[int],
):
    axes = giatools.default_normalized_axes
    image = giatools.Image.read(image_filepath, normalize_axes=axes)
    labelmap = giatools.Image.read(labelmap_filepath, normalize_axes=axes)

    # Establish compatibility of multi-channel/frame/etc. images with single-channel/frame/etc. label maps
    original_labelmap_shape = labelmap.shape
    for image_s, labelmap_s, (axis_idx, axis) in zip(image.shape, labelmap.shape, enumerate(axes)):
        if image_s > 1 and labelmap_s == 1 and axis not in 'YX':
            target_shape = list(labelmap.shape)
            target_shape[axis_idx] = image_s

            # Broadcast the image data to the target shape without copying
            if hasattr(labelmap.data, 'compute'):
                labelmap.data = da.broadcast_to(labelmap.data, target_shape)  # `data` is Dask array
            else:
                labelmap.data = np.broadcast_to(labelmap.data, target_shape, subok=True)  # `data` is NumPy array

    # Validate that the shapes of the images are compatible
    if image.shape != labelmap.shape:
        labelmap_shape_str = str(original_labelmap_shape)
        if labelmap.shape != original_labelmap_shape:
            labelmap_shape_str = f'{labelmap_shape_str}, broadcasted to {labelmap.shape}'
        raise ValueError(
            f'Shape mismatch between image {image.shape} and label map {labelmap_shape_str}, with {axes} axes.',
        )

    # Extract the image crops
    for label in giatools.image._unique(labelmap.data):
        if label in skip_labels:
            continue
        roi_mask = (labelmap.data == label)
        roi = crop_image_to_mask(image.data, roi_mask)
        roi_image = giatools.Image(roi, image.axes).normalize_axes_like(image.original_axes)
        roi_image.write(os.path.join(output_dir, f'{label}.{output_ext}'))


def crop_image_to_mask(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Crop the `data` array to the minimal bounding box in `mask`.

    The arguments are not modified.
    """
    assert data.shape == mask.shape

    # Crop `data` to the convex hull of the mask in each dimension
    for dim in range(data.ndim):
        mask1d = mask.any(axis=tuple(i for i in range(mask.ndim) if i != dim))
        mask1d_indices = np.where(mask1d)[0]

        # Convert `mask1d_indices` to a NumPy array if it is a Dask array
        if hasattr(mask1d_indices, 'compute'):
            mask1d_indices = mask1d_indices.compute()

        mask1d_indices_cvxhull = np.arange(min(mask1d_indices), max(mask1d_indices) + 1)

        # Crop the `data` to the minimal bounding box
        if hasattr(data, 'compute'):
            data = da.take(data, axis=dim, indices=mask1d_indices_cvxhull)  # `data` is a Dask array
        else:
            data = data.take(axis=dim, indices=mask1d_indices_cvxhull)  # `data` is a NumPy array

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
        skip_labels=frozenset(
            int(label.strip()) for label in args.skip_labels.split(',') if label.strip()
        ) if args.skip_labels.strip() else frozenset(),
    )
