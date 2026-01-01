import giatools
import libcarna
import libcarna._imshow

from giatools__0_7_2__cli__ToolBaseplate import parse_args  # noqa: I202

# Fail early if an optional backend is not available
giatools.require_backend('omezarr')

# Patch `libcarna._imshow` to return plain HTML
libcarna._imshow.IPythonHTML = lambda html: html


GEOMETRY_TYPE_INTENSITIES = 0
GEOMETRY_TYPE_MASK = 1


def wrap_color(params: dict) -> dict:
    """
    Return the `params` dictionary but wrap values for `color` with `libcarna.color`.
    """
    result = dict()
    for key, value in params.items():
        if key == 'color':
            value = libcarna.color(value)
        result[key] = value
    return result


if __name__ == "__main__":
    tool = giatools.ToolBaseplate()
    tool.add_input_image('intensities')
    tool.add_input_image('mask', required=False)
    tool.parser.add_argument('--html', type=str)
    parse_args(tool)  # TODO: Revert to `tool.parse_args` when 0.7.2 is on Conda

    # Validate the input image(s)
    try:
        for image in tool.args.input_images.values():
            if any(image.shape[image.axes.index(axis)] > 1 for axis in image.axes if axis not in 'ZYX'):
                raise ValueError(f'This tool is not applicable to images with {image.original_axes} axes.')

        # Create and configure frame renderer
        mode = getattr(libcarna, tool.args.params['mode'])(
            GEOMETRY_TYPE_INTENSITIES,
            sr=tool.args.params['sample_rate'],
            **tool.args.params['mode_kwargs']
        )
        mask_renderer = libcarna.mask_renderer(
            GEOMETRY_TYPE_MASK,
            sr=tool.args.params['sample_rate'],
            **wrap_color(tool.args.params['mask_renderer_kwargs']),
        )
        r = libcarna.renderer(
            tool.args.params['width'],
            tool.args.params['height'],
            [mode, mask_renderer],
        )

        # Build the scene graph
        root = libcarna.node()
        intensities = tool.args.input_images['intensities']
        intensities_volume = libcarna.volume(
            GEOMETRY_TYPE_INTENSITIES,
            intensities.normalize_axes_like(tool.args.params['axes']).data,
            parent=root,
            spacing=(
                intensities.metadata.pixel_size[0],
                intensities.metadata.z_spacing,
                intensities.metadata.pixel_size[1],
            ),
            normals=(tool.args.params['mode'] == 'dvr'),
        )
        camera = libcarna.camera(
            parent=root,
        ).frustum(
            **tool.args.params['camera']['kwargs'],
        ).translate(
            z=tool.args.params['camera']['distance'],
        )
        if (mask := tool.args.input_images.get('mask')):
            libcarna.volume(
                GEOMETRY_TYPE_MASK,
                mask.normalize_axes_like(tool.args.params['axes']).data,
                parent=intensities_volume,
                spacing=intensities_volume.spacing,
            )

        # Apply colormap
        if tool.args.params['colormap'] != 'custom':
            cmap_kwargs = dict()
            if (ramp_params := tool.args.params['ramp']):
                ramp_values = list()
                for normalized, value in (
                    (ramp_params['start_normalized'], ramp_params['start_value']),
                    (ramp_params['end_normalized'], ramp_params['end_value']),
                ):
                    ramp_values.append(
                        value if normalized else intensities_volume.normalized(value),
                    )
                cmap_kwargs['ramp'] = tuple(ramp_values)
            mode.cmap(tool.args.params['colormap'], **cmap_kwargs)
        else:
            pass  # TODO: implement and add test

        # Render
        html = libcarna.imshow(
            libcarna.animate(
                libcarna.animate.rotate_local(camera),
                n_frames=tool.args.params['video']['frames'],
            ).render(r, camera),
            mode.cmap.bar(intensities_volume),
            fps=tool.args.params['video']['fps'],
        )

        # Write the result
        with open(tool.args.raw_args.html, 'w') as fhtml:
            fhtml.write(html)

    except ValueError as err:
        exit(err.args[0])
