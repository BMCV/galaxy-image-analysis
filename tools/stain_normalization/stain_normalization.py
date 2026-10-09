import giatools
import numpy as np

# Fail early if an optional backend is not available
giatools.require_backend('omezarr')


if __name__ == "__main__":
    tool = giatools.ToolBaseplate()
    tool.add_input_image('input')
    tool.add_output_image('output')
    tool.add_output_image('output_h', required=False)
    tool.add_output_image('output_e', required=False)

    # The reference image is read separately, because it can have a different size than the input image
    tool.parser.add_argument('--reference', type=str, default=None)

    # Do not write the optional outputs which were not requested
    args = tool.parse_args()
    args.output_filepaths = {key: path for key, path in args.output_filepaths.items() if path is not None}

    # Drop the alpha channel, if any (the output image is allocated like the input image, so this must happen upfront)
    image = args.input_images['input']
    channel_axis = image.axes.index('C')
    if image.data.shape[channel_axis] == 4:
        args.input_images['input'] = giatools.Image(
            data=np.take(image.data, [0, 1, 2], axis=channel_axis),
            axes=image.axes,
            metadata=image.metadata,
            original_axes=image.original_axes,
        )

    for section in tool.run(
        'YXC',  # process the RGB image as a whole
        output_dtype_hint='preserve',
    ):
        section['output'] = section['input'].data
