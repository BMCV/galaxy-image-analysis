import giatools
import numpy as np

# Fail early if an optional backend is not available
giatools.require_backend('omezarr')


if __name__ == "__main__":
    tool = giatools.ToolBaseplate()
    tool.add_input_image('input')
    tool.parser.add_argument('--html', type=str)
    tool.parse_args()

    # Validate the input image
    try:
        image = tool.args.input_images['input']
        if any(image.shape[image.axes.index(axis)] > 1 for axis in image.axes if axis not in 'ZYX'):
            raise ValueError(f'This tool is not applicable to images with {image.original_axes} axes.')

        with open(tool.args.raw_args.html, 'w') as fhtml:
            fhtml.write(
'<html>'
'<body>'
'<span style="font-family: sans-serif;">sans</span> serif'
'</body>'
'</html>'
            )

    except ValueError as err:
        exit(err.args[0])
