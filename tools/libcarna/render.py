import giatools
import libcarna
import libcarna._imshow
import pandas as pd

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
    tool.parser.add_argument('--colormap', type=str)
    tool.parser.add_argument('--html', type=str)
    tool.parse_args()

    # Load custom colormap
    if tool.args.raw_args.colormap:
        df_colormap = pd.read_csv(tool.args.raw_args.colormap, delimiter='\t')

    # Validate the input image(s)
    try:
        for image in tool.args.input_images.values():
            if any(image.shape[image.axes.index(axis)] > 1 for axis in image.axes if axis not in 'ZYX'):
                raise ValueError(f'This tool is not applicable to images with {image.original_axes} axes.')

        # Create and configure frame renderer
        print('Sample rate:', tool.args.params['sample_rate'])
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
        print('EGL Vendor:', r.gl_context.vendor)

        # Build the scene graph
        root = libcarna.node()
        intensities = tool.args.input_images['intensities']
        intensities_volume = libcarna.volume(
            GEOMETRY_TYPE_INTENSITIES,
            intensities.normalize_axes_like(tool.args.params['axes']).data,
            parent=root,
            units=tool.args.params['units'],
            spacing=[
                {
                    'X': intensities.metadata.pixel_size[0] or 1,
                    'Y': intensities.metadata.pixel_size[1] or 1,
                    'Z': intensities.metadata.z_spacing or 1,
                }
                [axis] for axis in tool.args.params['axes']
            ],
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
        if tool.args.params['colormap'] == 'custom':
            mode.cmap.clear()
            i0, color0 = None, None
            for row in df_colormap.to_dict(orient='records'):
                match row['type']:
                    case 'relative':
                        i1 = row['intensity']
                    case 'absolute':
                        i1 = intensities_volume.normalized(row['intensity'])
                    case _:
                        raise ValueError('Unknown intensity type: "{}"'.format(row['type']))
                color1 = libcarna.color(row['color'])
                if i0 is not None:
                    mode.cmap.linear_segment(i0, i1, color0, color1)
                i0, color0 = i1, color1
        else:
            cmap_kwargs = dict()
            if (ramp_params := tool.args.params['ramp']):
                ramp_values = list()
                for val_type, value in (
                    (ramp_params['start_type'], ramp_params['start_value']),
                    (ramp_params['end_type'], ramp_params['end_value']),
                ):
                    ramp_values.append(
                        value if val_type == 'relative' else intensities_volume.normalized(value),
                    )
                cmap_kwargs['ramp'] = tuple(ramp_values)
            mode.cmap(tool.args.params['colormap'], **cmap_kwargs)

        # Render
        colorbars = list()
        if tool.args.params['colorbar']:
            colorbars.append(mode.cmap.bar(intensities_volume))
        html = libcarna.imshow(
            libcarna.animate(
                libcarna.animate.rotate_local(camera),
                n_frames=tool.args.params['video']['frames'],
            ).render(r, camera),
            *colorbars,
            fps=tool.args.params['video']['fps'],
        )

        # Write the result
        with open(tool.args.raw_args.html, 'w') as fhtml:
            fhtml.write(html)

    except ValueError as err:
        exit(err.args[0])
