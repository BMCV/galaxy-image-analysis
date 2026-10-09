import os
import xml.etree.ElementTree as ET

import giatools
import numpy as np
import tifffile
from torchstain.base.normalizers import MacenkoNormalizer, ReinhardNormalizer

# Fail early if an optional backend is not available
giatools.require_backend('omezarr')

# Display names of the methods
METHOD_NAMES = {
    'macenko': 'Macenko',
    'reinhard': 'Reinhard',
    'modified_reinhard': 'modified Reinhard',
}

# The covariance of the stained pixels must have two non-trivial principal directions (i.e., at least three pixels)
MIN_STAINED_PIXELS = 3


def load_reference(filepath, verbose):
    """
    Load the reference image, which is read separately because it can have a different size than the input image.
    """
    image = giatools.Image.read(filepath)
    if verbose:
        print(f'[reference] Input image axes: {image.original_axes}')
        print(f'[reference] Input image shape: {image.original_shape}')
        print(f'[reference] Input image dtype: {image.data.dtype}')
        if (metadata_str := str(image.metadata)):
            print(f'[reference] Input image {metadata_str}')

    if any(size > 1 for axis, size in zip(image.axes, image.data.shape) if axis not in 'YXC'):
        exit('The reference image must be a 2-D single-frame image.')

    image = image.squeeze_like('YXC')
    if 'C' not in image.axes or image.data.shape[-1] not in (3, 4):
        exit('The reference image must be an RGB or RGBA image.')
    return image.data[..., :3]  # drop the alpha channel, if any


def require_8bit(name, image):
    if image.dtype != np.uint8:
        exit(f'The {name} must have 8-bit pixel values (but has dtype {image.dtype}).')


def count_stained_pixels(image, Io, beta):
    """
    Count the pixels which are used by the Macenko method to estimate the stain vectors (i.e., which are not
    transparent, meaning that their optical density is not below `beta` in any channel).
    """
    optical_density = -np.log((image.reshape((-1, 3)).astype(float) + 1) / Io)
    return int((~(optical_density < beta).any(axis=1)).sum())


def fail_macenko(name, image, Io, beta):
    num_pixels = image.shape[0] * image.shape[1]
    num_stained = count_stained_pixels(image, Io, beta)
    exit(
        f'Macenko normalization failed for the {name}: Only {num_stained} of {num_pixels} pixels '
        f'({100 * num_stained / num_pixels:.1f}%) have an optical density of at least beta={beta:g} in all channels, '
        'so the stain vectors cannot be estimated. The image is probably mostly background or too bright. '
        'Use an image with more tissue, or lower the advanced parameter beta.'
    )


def macenko(name, image, Io, beta, func):
    """
    Call `func` (which performs the Macenko estimation on `image`) and fail with a clear message if that is not
    possible.
    """
    if count_stained_pixels(image, Io, beta) < MIN_STAINED_PIXELS:
        fail_macenko(name, image, Io, beta)
    try:
        return func()
    except (np.linalg.LinAlgError, ValueError, IndexError):
        fail_macenko(name, image, Io, beta)


def write_pixel_size_to_ome_xml(filepath, metadata):
    """
    Add the pixel size to the OME-XML of an OME-TIFF file, because `giatools` does not write it.

    This is a self-contained workaround. To remove it, delete this function, its call below (as well as the `ET` and
    `tifffile` imports), and the corresponding assertions in the tests.
    """
    ome_namespace = 'http://www.openmicroscopy.org/Schemas/OME/2016-06'

    # Maps the units of `giatools` to the units of the OME schema
    ome_units = {
        'nm': 'nm',
        'um': 'µm',
        'mm': 'mm',
        'cm': 'cm',
        'm': 'm',
        'km': 'km',
        'inch': 'in',
    }

    if metadata.resolution is None or metadata.unit is None:
        print('No pixel size (with unit) available, not writing it to the OME-XML')
        return
    print(f'Writing pixel size {metadata.pixel_size} with unit {metadata.unit!r} to the OME-XML')

    ET.register_namespace('', ome_namespace)
    ET.register_namespace('xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    ome_xml = ET.fromstring(tifffile.tiffcomment(filepath))
    pixels = ome_xml.find(f'.//{{{ome_namespace}}}Pixels')
    for axis, size in zip('XY', metadata.pixel_size):
        pixels.set(f'PhysicalSize{axis}', repr(float(size)))
        pixels.set(f'PhysicalSize{axis}Unit', ome_units[metadata.unit])
    # TIFF strings must be 7-bit ASCII, so non-ASCII characters (like the "micro" in "µm") are written as XML entities
    tifffile.tiffcomment(filepath, comment=ET.tostring(ome_xml, encoding='us-ascii', xml_declaration=False).decode('ascii'))


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
    params = args.params
    method = params['method']

    # Drop the alpha channel, if any (the output image is allocated like the input image, so this must happen upfront)
    image = args.input_images['input']
    channel_axis = image.axes.index('C')
    if image.data.shape[channel_axis] == 4:
        args.input_images['input'] = giatools.Image(
            data=image.data[(slice(None),) * channel_axis + (slice(0, 3),)],
            axes=image.axes,
            metadata=image.metadata,
            original_axes=image.original_axes,
        )
    require_8bit('H&E image', args.input_images['input'].data)

    # Load the reference image and set up the normalizer
    reference = None
    if (reference_filepath := tool.args.raw_args.reference) is not None:
        reference = np.asarray(load_reference(reference_filepath, args.verbose))
        require_8bit('reference image', reference)

    if method == 'macenko':
        Io, alpha, beta = params['Io'], params['alpha'], params['beta']
        normalizer = MacenkoNormalizer(backend='numpy')
        if reference is not None:
            print('Fitting the Macenko normalizer to the reference image')
            macenko('reference image', reference, Io, beta, lambda: normalizer.fit(reference, Io=Io, alpha=alpha, beta=beta))
        else:
            print('Using the built-in H&E reference of the Macenko normalizer')
    else:
        if reference is None:
            exit(f'The {METHOD_NAMES[method]} method requires a reference image.')
        normalizer = ReinhardNormalizer(backend='numpy', method=None if method == 'reinhard' else 'modified')
        print(f'Fitting the {METHOD_NAMES[method]} normalizer to the reference image')
        normalizer.fit(reference)

    # Let `giatools` write OME-TIFF files (which it does if the filename ends with `.ome.tiff`)
    final_filepaths = dict(args.output_filepaths)
    if params['output_format'] == 'ome.tiff':
        for key, filepath in final_filepaths.items():
            args.output_filepaths[key] = os.path.splitext(filepath)[0] + '.ome.tiff'

    for section in tool.run(
        'YXC',  # process the RGB image as a whole
        output_dtype_hint='preserve',
        output_h_dtype_hint='preserve',
        output_e_dtype_hint='preserve',
    ):
        image = np.asarray(section['input'].data)  # conversion is required in case of a Dask array (OME-Zarr)
        print(f'Applying {METHOD_NAMES[method]} normalization')

        if method == 'macenko':
            section['output'], h, e = macenko(
                'H&E image', image, Io, beta,
                lambda: normalizer.normalize(image, Io=Io, alpha=alpha, beta=beta, stains='output_h' in final_filepaths),
            )
            if 'output_h' in final_filepaths:
                section['output_h'] = h
                section['output_e'] = e
        else:
            section['output'] = normalizer.normalize(image)

    # Move the OME-TIFF files to the expected location
    if params['output_format'] == 'ome.tiff':
        for key, final_filepath in final_filepaths.items():
            write_pixel_size_to_ome_xml(args.output_filepaths[key], tool.processor.outputs[key].metadata)  # workaround
            os.replace(args.output_filepaths[key], final_filepath)
