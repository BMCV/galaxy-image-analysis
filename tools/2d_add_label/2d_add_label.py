import giatools
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

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


#def hex_rgba_to_uint8(hex_str):
#    hex_str = hex_str.lstrip('#')
#
#    r = int(hex_str[0:2], 16)
#    g = int(hex_str[2:4], 16)
#    b = int(hex_str[4:6], 16)
#    a = int(hex_str[6:8], 16)
#
#    return (r, g, b, a)


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
        #for section in tool.run('XYC'):  # the validation code above guarantees that we will have only a single iteration

        # TODO: expose as tool parameters
        fp_lower = 'min'
        fp_upper = 'max'
        label_text_color = '#000000ff'
        label_background_color = '#ffffffff'
        fontsize = 12
        position_v = tool.args.params['position_v']
        position_h = tool.args.params['position_h']

        # Create RGBA8 working copy of the input image
        #img = get_rgba8_copy(section['input_image'].data, fp_lower, fp_upper)
        img = get_rgba8_copy(
            tool.args.input_images['input_image'].normalize_axes_like('XYC').data,
            fp_lower,
            fp_upper,
        )

        # Determine the label height in pixels
        label_height = (fontsize * 100) // 72

        # Extend the image above/below, if required
        if position_v in ('above', 'below'):
            label_bg_color = np.multiply(255, matplotlib.colors.to_rgba(label_background_color)).astype(np.uint8)
            label_bg = np.full((label_height, img.shape[1], 4), 1, dtype=np.uint8) * label_bg_color
            blocks = (img, label_bg)
            if position_v == 'above':
                blocks = blocks[::-1]
            img = np.concatenate(blocks, axis=0)
            label_background_color = '#00000000'

        # Determine the vertical text alignment and coordinate
        match position_v:
            case 'above':
                position_v = 'top'
                text_y = 0
            case 'below':
                position_v = 'bottom'
                text_y = img.shape[0] + label_height - 1
            case 'top':
                text_y = 0
            case 'bottom':
                text_y = img.shape[0] - 1

        # Determine the horizontal text coordinate
        match position_h:
            case 'left':
                text_x = 0
            case 'right':
                text_x = img.shape[1] - 1
            case 'center':
                text_x = (img.shape[1] - 1) / 2

        # Create figure
        fig = plt.figure(figsize=np.divide(img.shape[:2][::-1], 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(img)

        # Paint text
        ax.text(
            text_x,
            text_y,
            str(tool.args.params['label_text']),
            color=label_text_color,
            backgroundcolor=label_background_color,
            ha=position_h,
            va=position_v,
        )
        #fig.canvas.draw()

        # Determine the result image
        #img = np.array(fig.canvas.renderer._renderer)
        #section['output_image'] = img

        # Write the result image
        fig.canvas.print_png(tool.args.raw_args.output_image)

    except ValueError as err:
        #exit(err.args[0])
        raise
