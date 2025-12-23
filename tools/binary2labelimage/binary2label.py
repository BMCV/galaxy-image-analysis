import argparse
import json

import giatools
import numpy as np
import scipy.ndimage as ndi

# Fail early if an optional backend is not available
giatools.require_backend('omezarr')


def label_watershed(arr: np.ndarray, **kwargs) -> np.ndarray:
    import skimage.util
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    distance = ndi.distance_transform_edt(arr)
    local_max_indices = peak_local_max(
        distance,
        labels=arr,
        **kwargs,
    )
    local_max_mask = np.zeros(arr.shape, dtype=bool)
    local_max_mask[tuple(local_max_indices.T)] = True
    markers = ndi.label(local_max_mask)[0]
    res = watershed(-distance, markers, mask=arr)
    return skimage.util.img_as_uint(res)  # converts to uint16


if __name__ == '__main__':

    # Parse CLI parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input file path')
    parser.add_argument('output', type=str, help='Output file path (TIFF)')
    parser.add_argument('params', type=str)
    args = parser.parse_args()

    # Read the config file
    with open(args.params) as cfgf:
        cfg = json.load(cfgf)

    # Read the input image and ensure that it is truly binary
    image = giatools.Image.read(args.input)
    image.data = (image.data > 0)
    print('Input image shape:', image.data.shape)
    print('Input image axes:', image.axes)
    print('Input image dtype:', image.data.dtype)

    # Validate the input image and the selected method
    try:
        if (method := cfg.pop('method')) == 'watershed' and image.data.shape[image.axes.index('Z')] > 1:
            raise ValueError(f'Method "{method}" is not applicable to 3-D images.')

        elif image.data.shape[image.axes.index('C')] > 1:
            raise ValueError('Multi-channel images are forbidden to avoid confusion with multi-channel labels (e.g., RGB labels).')

        else:

            # Perform the labeling
            result = np.empty(image.data.shape, np.uint16)
            match method:

                case 'cca':
                    for sl, section in image.iterate_jointly('ZYX'):
                        result[sl] = ndi.label(section, **cfg)[0].astype(result.dtype)

                case 'watershed':
                    for sl, section in image.iterate_jointly('YX'):
                        result[sl] = label_watershed(section, **cfg)  # already uint16

                case _:
                    raise ValueError(f'Unknown method: "{method}"')

            # Write the result image
            image.data = result
            image = image.normalize_axes_like(image.original_axes)
            print('Output image shape:', image.data.shape)
            print('Output image axes:', image.axes)
            image.write(
                args.output,
                backend='tifffile',
            )

    # Exit and print error to stderr
    except ValueError as err:
        exit(err.args[0])
