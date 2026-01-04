import giatools
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import skimage.measure

# Fail early if an optional backend is not available
giatools.require_backend('omezarr')


def surface(labels: np.ndarray, label: int) -> int:
    """
    Ad-hoc implementation for computation of the "perimeter" of an object in 3D (that is a surface).
    """
    assert labels.ndim == 3  # sanity check

    # Create 3-D structuring element with 4-connectivity
    selem = np.zeros((3, 3, 3), bool)
    for ijk in np.ndindex(*selem.shape):
        if (np.array(ijk) == 1).sum() >= 2:
            selem[*ijk] = True   # noqa: E999
    assert selem.sum() == 7  # sanity check

    # Compute the area of the surface
    cc = (labels == label)
    cc_interior = ndi.binary_erosion(cc, selem)
    surface = np.logical_xor(cc, cc_interior)
    return surface.sum()  # number of voxels on the surface of the object


def compute_if_dask(obj):
    """
    Return the computed object or array if it is a Dask array or deferred computable Dask object.
    """
    return obj.compute() if hasattr(obj, 'compute') else obj


if __name__ == '__main__':
    tool = giatools.ToolBaseplate()
    tool.add_input_image('labels')
    tool.add_input_image('intensities', required=False)
    tool.parser.add_argument('--output', type=str)
    tool.parse_args()

    # Validate the input image
    try:
        label_image = tool.args.input_images['labels']
        if any(label_image.shape[label_image.axes.index(axis)] > 1 for axis in label_image.axes if axis not in 'ZYX'):
            raise ValueError(f'This tool is not applicable to images with {label_image.original_axes} axes.')

        # Extract the image features
        for section in tool.run('ZYX'):  # the validation code above guarantees that we will have only a single iteration
            df = pd.DataFrame()

            # Get the labels array and cast to `uint8` if it is `bool` (`skimage.measure.regionprops` refuses `bool` typed arrays)
            labels_section_data = section['labels'].data.squeeze()
            if np.issubdtype(labels_section_data.dtype, bool):
                print('Convert labels from bool to uint8')
                labels_section_data = labels_section_data.astype(np.uint8)

            # Some features currently cannot be computed from Dask arrays
            if any(
                feature_name in tool.args.params['features'] for feature_name in (
                    'inertia_tensor_eigvals',
                    'axis_major_length',
                    'axis_minor_length',
                    'eccentricity',
                    'orientation',
                    'moments_hu',
                )
            ):
                labels_section_data = compute_if_dask(labels_section_data)

            # Compute the image features
            if 'intensities' in tool.args.input_images:
                regions = skimage.measure.regionprops(labels_section_data, intensity_image=section['intensities'].data.squeeze())
            else:
                regions = skimage.measure.regionprops(labels_section_data, intensity_image=None)
            df['it'] = np.arange(len(regions))
            for feature_name in tool.args.params['features']:

                # Add the object label
                if feature_name == 'label':
                    df['label'] = df['it'].map(lambda ait: regions[ait].label)

                # Add the object perimeter/surface
                elif feature_name == 'perimeter' and labels_section_data.ndim == 3:
                    df['perimeter'] = df['it'].map(
                        lambda ait: surface(labels_section_data, regions[ait].label),  # `skimage.measure.regionprops` cannot compute perimeters for 3-D data
                    )

                # Skip features that are not available when processing 3-D images
                elif feature_name in ('eccentricity', 'moments_hu', 'orientation') and labels_section_data.ndim == 3:
                    print(f'Skip feature that is not available for 3-D images: "{feature_name}"')

                # Add another feature from `regions` that was computed via `skimage.measure.regionprops`
                else:
                    try:
                        df[feature_name] = df['it'].map(lambda ait: getattr(regions[ait], feature_name))
                    except TypeError:
                        raise ValueError(f'Unknown feature: "{feature_name}"')

            # Resolve any remaining Dask objects to the actual values (e.g., when processing Zarrs)
            df = df.map(compute_if_dask)

            # Convert lists/tuples/arrays to lists of plain Python numbers (e.g., float instead of np.float64)
            df = df.map(
                lambda obj: np.asarray(obj).tolist() if type(obj) in (list, tuple, np.ndarray) else obj,
            )

            del df['it']
            df.to_csv(tool.args.raw_args.output, sep='\t', lineterminator='\n', index=False)

    except ValueError as err:
        exit(err.args[0])
