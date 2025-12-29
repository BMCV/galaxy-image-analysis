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

    tool = giatools.ToolBaseplate()
    tool.add_input_image('input')
    tool.add_output_image('output')
    tool.parse_args()

    # Validate the input image and the selected method
    try:
        input_image = tool.args.input_images['input']
        if (method := tool.args.params.pop('method')) == 'watershed' and input_image.shape[input_image.axes.index('Z')] > 1:
            raise ValueError(f'Method "{method}" is not applicable to 3-D images.')

        elif input_image.shape[input_image.axes.index('C')] > 1:
            raise ValueError('Multi-channel images are forbidden to avoid confusion with multi-channel labels (e.g., RGB labels).')

        else:

            # Choose the requested labeling method
            match method:

                case 'cca':
                    joint_axes = 'ZYX'
                    
                    def label(input_section_bin):
                        return ndi.label(input_section_bin, **tool.args.params)[0].astype(np.uint16)

                case 'watershed':
                    joint_axes = 'YX'
                    
                    def label(input_section_bin):
                        return label_watershed(input_section_bin, **tool.args.params)  # already uint16

                case _:
                    raise ValueError(f'Unknown method: "{method}"')

        # Perform the labeling
        for section in tool.run(joint_axes):
            section['output'] = label(
                section['input'].data > 0,  # ensure that the input data is truly binary
            )

    # Exit and print error to stderr
    except ValueError as err:
        exit(err.args[0])
