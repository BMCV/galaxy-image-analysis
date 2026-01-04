"""
Copyright 2017-2025 Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
"""

import giatools
import numpy as np
import skimage.filters
import skimage.util

# Fail early if an optional backend is not available
giatools.require_backend('omezarr')


class DefaultThresholdingMethod:

    def __init__(self, thres, **kwargs):
        self.thres = thres
        self.kwargs = kwargs

    def __call__(self, image, *args, offset=0, **kwargs):
        thres = self.thres(image, *args, **(self.kwargs | kwargs))
        return image > thres + offset

    def __str__(self):
        return self.thres.__name__


class ManualThresholding:

    def __call__(self, image, threshold1: float, threshold2: float | None, **kwargs):
        if threshold2 is None:
            return image > threshold1
        else:
            threshold1, threshold2 = sorted((threshold1, threshold2))
            return skimage.filters.apply_hysteresis_threshold(image, threshold1, threshold2)

    def __str__(self):
        return 'Manual'


methods = {
    'manual': ManualThresholding(),

    'otsu': DefaultThresholdingMethod(skimage.filters.threshold_otsu),
    'li': DefaultThresholdingMethod(skimage.filters.threshold_li),
    'yen': DefaultThresholdingMethod(skimage.filters.threshold_yen),
    'isodata': DefaultThresholdingMethod(skimage.filters.threshold_isodata),

    'loc_gaussian': DefaultThresholdingMethod(skimage.filters.threshold_local, method='gaussian'),
    'loc_median': DefaultThresholdingMethod(skimage.filters.threshold_local, method='median'),
    'loc_mean': DefaultThresholdingMethod(skimage.filters.threshold_local, method='mean'),
}


if __name__ == "__main__":
    tool = giatools.ToolBaseplate()
    tool.add_input_image('input')
    tool.add_output_image('output')
    tool.parse_args()

    # Retrieve general parameters
    method = tool.args.params.pop('method')
    invert = tool.args.params.pop('invert')

    # Perform thresholding
    method_impl = methods[method]
    print(
        'Thresholding:',
        str(method_impl),
        'with',
        ', '.join(
            f'{key}={repr(value)}' for key, value in (tool.args.params | dict(invert=invert)).items()
        ),
    )
    for section in tool.run('ZYX', output_dtype_hint='binary'):
        section_output = method_impl(
            image=np.asarray(section['input'].data),  # some implementations have issues with Dask arrays
            **tool.args.params,
        )
        if invert:
            section_output = np.logical_not(section_output)
        section['output'] = section_output
