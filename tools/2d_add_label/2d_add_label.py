import giatools
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import skimage.color

# Fail early if an optional backend is not available
giatools.require_backend('omezarr')


def get_rgba8_copy(img, fp_lower, fp_upper):
    img = np.squeeze(img)
    assert img.ndim == 2 or (img.ndim == 3 and img.shape[-1] in (3, 4))
    assert fp_lower == 'min' or abs(float(fp_lower)) < np.inf  # 'min' or number
    assert fp_upper == 'max' or abs(float(fp_upper)) < np.inf  # 'max' or number

    # Convert from floating point
    if str(img.dtype).startswith('float'):
        a = img.min() if fp_lower == 'min' else float(fp_lower)
        b = img.max() if fp_upper == 'max' else float(fp_upper)

        if a > b:
            raise ValueError(
                f'Lower bound ({a:g}) must be less than upper bound ({b:g}).'
            )
        if a == b:
            raise ValueError(
                'Floating point conversion is undefined (lower and upper bounds are identical).'
            )

        # Perform linear mapping to [0, 1]
        img = img.clip(a, b)
        img = (img - a) / (b - a)

        # Convert to uint8
        img = np.round(img * 255).astype(np.uint8)

    # Convert from uint16
    elif img.dtype == np.uint16:
        img = (img // 256).astype(np.uint8)

    # Other dtypes than float, uint8, uint16 are not supported
    elif img.dtype != np.uint8:
        raise ValueError(f'unknown dtype: {img.dtype}')

    if img.ndim == 2:  # single-channel –> RGB (propagate along C-axis)
        result = np.dstack([img] * 3)  # no copy required here, forcefully performed below
    if img.shape[2] == 3:  # 3-channel –> RGBA (append alpha channel)
        a = np.full(img.shape[:2], 0xff, dtype=np.uint8)
        result = np.concatenate([img, a[:, :, None]], axis=2).copy()
    else:  # 4-channel –> RGBA (just copy)
        result = img[:, :, :4].copy()
    assert result.dtype == np.uint8, result.dtype  # sanity check
    return result


if __name__ == "__main__":
    tool = giatools.ToolBaseplate()
    tool.add_input_image('input_image')
    tool.add_output_image('output_image')
    tool.parse_args()
    try:

        # Validate the input image
        input_image = tool.args.input_images['input_image']
        if any(input_image.shape[input_image.axes.index(axis)] > 1 for axis in input_image.axes if axis not in 'XYC'):
            raise ValueError(f'This tool is not applicable to images with {input_image.original_axes} axes.')
        if (n_channels := input_image.shape[input_image.axes.index('C')]) not in (1, 3, 4):
            raise ValueError(f'This tool is not applicable to images with {n_channels} channels.')

        # Extract the image features
        for section in tool.run('XYC'):  # the validation code above guarantees that we will have only a single iteration

            # Create RGBA8 working copy of the input image
            img_rgba8 = get_rgba8_copy(section['input_image'].data)

            # Determine the result image
            section['output_image'] = img_rgba8

    except ValueError as err:
        exit(err.args[0])
