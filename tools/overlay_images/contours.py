"""
Copyright (c) 2017-2023 Leonid Kostrykin, Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
"""

import numpy as np
import skimage.morphology as morph


class ContourPaint:
    """Yields masks corresponding to contours of objects.

    :param fg_mask: Binary mask of the image foreground. Any contour never overlaps the image foreground except of those image regions corresponding to the contoured object itself.
    :param thickness: The thickness of the contour (width, in pixels).
    :param where: The position of the contour (``inner``, ``center``, or ``outer``).
    """

    def __init__(self, fg_mask, thickness, where='center'):
        assert where in ('inner', 'center', 'outer')
        self.fg_mask = fg_mask
        self.where = where
        self.thickness = thickness
        if where == 'inner':
            self.selem_inner = morph.disk(self.thickness)
            self.selem_outer = None
        elif where == 'center':
            self.selem_inner = morph.disk(self.thickness - self.thickness // 2)
            self.selem_outer = morph.disk(self.thickness // 2)
        elif where == 'outer':
            self.selem_inner = None
            self.selem_outer = morph.disk(self.thickness)

    def get_contour_mask(self, mask):
        """Returns the binary mask of the contour of an object.

        :param mask: Binary mask of an object.
        :return: Binary mask of the contour
        """
        if self.selem_inner is not None:
            inner_contour = np.logical_xor(mask, morph.binary_erosion(mask, self.selem_inner))
        else:
            inner_contour = np.zeros(mask.shape, bool)
        if self.selem_outer is not None:
            outer_contour = np.logical_and(np.logical_not(self.fg_mask), morph.binary_dilation(mask, self.selem_outer))
        else:
            outer_contour = np.zeros(mask.shape, bool)
        return np.logical_or(inner_contour, outer_contour)
