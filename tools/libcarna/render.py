import giatools
import libcarna
import libcarna._imshow

# Fail early if an optional backend is not available
giatools.require_backend('omezarr')

# Patch `libcarna._imshow` to return plain HTML
libcarna._imshow.IPythonHTML = lambda html: html


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

        # Create and configure frame renderer
        GEOMETRY_TYPE_VOLUME = 0
        rs = getattr(libcarna, tool.args.params['mode'])(
            GEOMETRY_TYPE_VOLUME,
            sr=tool.args.params['sample_rate'],
            **tool.args.params['mode_kwargs']
        )
        r = libcarna.renderer(
            800,  # TODO: add parameter
            450,
            [rs],
        )

        # Build the scene graph
        root = libcarna.node()

        volume = libcarna.volume(
            GEOMETRY_TYPE_VOLUME,
            image.normalize_axes_like('XZY').data,
            parent=root,
            spacing=(
                image.metadata.pixel_size[0],
                image.metadata.z_spacing,
                image.metadata.pixel_size[1],
            ),
        )

        camera = libcarna.camera(
            parent=root,
        ).frustum(
            **tool.args.params['camera']['kwargs'],
        ).translate(
            z=tool.args.params['camera']['distance'],
        )

        # Render
        html = libcarna.imshow(
            libcarna.animate(
                libcarna.animate.rotate_local(camera),
                n_frames=tool.args.params['video']['frames'],
            ).render(r, camera),
            rs.cmap.bar(volume),
            fps=tool.args.params['video']['fps'],
        )

        # Write the result
        with open(tool.args.raw_args.html, 'w') as fhtml:
            fhtml.write(html)

    except ValueError as err:
        exit(err.args[0])
