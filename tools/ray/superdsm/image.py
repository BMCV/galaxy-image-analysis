from .io import imread

import numpy as np


def get_pixel_map(shape, normalized=False):
    """Returns two 2D arrays corresponding to pixel coordinates of an array of the given shape.

    The first array corresponds to the row indices (coordinates), the second array corresponds to the column indices (coordinates). The coordinates are normalized to the range between 0 and 1 if ``normalized`` is ``True``.

    :return: The two 2D arrays, encapsulated as a single 3D array.

    .. runblock:: pycon

       >>> import superdsm.image
       >>> superdsm.image.get_pixel_map((6, 3))
       >>> superdsm.image.get_pixel_map((6, 3), normalized=True)
    """
    z = (np.array(shape) - 1. if normalized else np.ones(2))[Ellipsis, None, None]
    z[z == 0] = 1
    return np.indices(shape) / z


def bbox(mask, include_end=False):
    """Returns the bounding box of a mask.

    :param include_end: If ``True``, then the pair of last indices ``bbox[0][1]`` and ``bbox[1][1]`` is *included* in the specified ranges. It is *excluded* otherwise.
    :return: Tuple ``(bbox, sel)``, where ``bbox[0]`` are the first and last indices of the rows and ``bbox[1]`` are the first and last indices of the columns, and ``sel`` is a numpy slice corresponding to that image region.

    .. runblock:: pycon

       >>> import superdsm.image
       >>> import numpy as np
       >>> mask = np.array([[0, 0, 0, 0, 0],
       ...                  [0, 0, 0, 1, 0],
       ...                  [0, 0, 1, 1, 0],
       ...                  [0, 0, 1, 0, 0]])
       >>> superdsm.image.bbox(mask.astype(bool))
       >>> superdsm.image.bbox(mask.astype(bool), include_end=True)
    """
    mask_a0 = mask.any(axis=0)
    mask_a1 = mask.any(axis=1)
    ret = np.array([np.where(mask_a1)[0][[0, -1]], np.where(mask_a0)[0][[0, -1]]])
    if not include_end: ret += np.array([0, 1])
    return ret, np.s_[ret[0][0] : ret[0][1], ret[1][0] : ret[1][1]]


def normalize_image(img):
    """Normalizes the image intensities to the range from 0 to 1.

    The original image ``img`` is not modified.

    :return: The normalized image.
    """
    img_diff = img.max() - img.min()
    if img_diff == 0: img_diff = 1
    return (img - img.min()).astype(float) / img_diff

class Image:
    """This class facilitates the work with images, image masks, and image regions.
    """

    def __init__(self, model=None, mask=None, full_mask=None, offset=(0,0)):
        self.model     = model
        self.mask      = mask if mask is not None else np.ones(model.shape, bool)
        self.full_mask = full_mask if full_mask is not None else self.mask
        self.offset    = offset

    def shrink_mask(self, mask):
        """Reduces a mask so it can be used to access this image.
        """
        return mask[self.offset[0] : self.offset[0] + self.mask.shape[0],
                    self.offset[1] : self.offset[1] + self.mask.shape[1]]

    def get_region(self, mask, shrink=False):
        """Returns the image region specified by a mask.
        
        :param mask: The binary mask used to specify the image region.
        :param shrink: If ``True``, the image region will be reduced.
        :return: The image region specified by the mask.
        """
        mask = np.logical_and(self.mask, mask)
        if shrink:
            _bbox = bbox(mask)
            return Image(self.model[_bbox[1]], mask[_bbox[1]], full_mask=mask, offset=tuple(_bbox[0][:,0]))
        else:
            return Image(self.model, mask)
    
    @staticmethod
    def create_from_array(img, mask=None, normalize=True):
        """Creates an instance from an image and a mask. 
        """
        assert mask is None or (isinstance(mask, np.ndarray) and mask.dtype == bool)
        if normalize: img = normalize_image(img)
        return Image(model=img, mask=mask)

    def get_map(self, normalized=True, pad=0):
        """Returns two 2D arrays corresponding to the pixel coordinates.

        See :py:meth:`~.get_pixel_map` for details.
        """
        assert pad >= 0 and isinstance(pad, int)
        return get_pixel_map(np.add(self.model.shape, 2 * pad), normalized)
