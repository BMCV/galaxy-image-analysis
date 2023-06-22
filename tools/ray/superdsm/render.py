from ._aux import render_objects_foregrounds

import numpy as np
import warnings, math

from skimage import morphology, segmentation
from scipy   import ndimage

import skimage.draw
import matplotlib.pyplot as plt


def draw_line(p1, p2, thickness, shape):
    """Returns binary mask corresponding to a straight line.

    :param p1: Coordinates of the first endpoint of the line.
    :param p2: Coordinates of the second endpoint of the line.
    :param thickness: The thickness of the line.
    :param shape: The shape of the binary mask to be returned.
    :return: Binary mask corresponding to the straight line between the two endpoints.
    """
    assert thickness >= 1
    threshold = (thickness + 1) / 2
    if np.allclose(threshold, round(threshold)):
        box = np.array((np.min((p1, p2), axis=0), np.max((p1, p2), axis=0)))
        n = math.ceil(threshold) - 1
        box[0] -= n
        box[1] += n
        box = box.clip(0, np.subtract(shape, 1))
        buf = np.zeros(1 + box[1] - box[0])
        p1  = p1 - box[0]
        p2  = p2 - box[0]
        rr, cc = skimage.draw.line(*p1, *p2)
        buf[rr, cc] = 1
        buf = ndimage.distance_transform_edt(buf == 0) < threshold
        result = np.zeros(shape)
        result[box[0,0] : box[1,0] + 1, box[0,1] : box[1,1] + 1] = buf
        return result
    else:
        thickness1 = 2 * int((thickness + 1) // 2) - 1
        thickness2 = thickness1 + 2
        buf1 = draw_line(p1, p2, thickness1, shape)
        buf2 = draw_line(p1, p2, thickness2, shape)
        return (buf2 * (thickness - thickness1) / (thickness2 - thickness1) + buf1).clip(0, 1)


def render_adjacencies(data, normalize_img=True, edge_thickness=3, endpoint_radius=5, endpoint_edge_thickness=2,
                       edge_color=(1,0,0), endpoint_color=(1,0,0), endpoint_edge_color=(0,0,0), override_img=None):
    """Returns a visualization of the adjacency graph (see :ref:`pipeline_theory_c2freganal`).

    By default, the adjacency graph is rendered on top of the contrast-enhanced raw image itensities. Contrast enhancement is performed using the :py:meth:`normalize_image` function.

    :param data: The pipeline data object.
    :param normalize_img: ``True`` if contrast-enhancement should be performed and ``False`` otherwise. Only used if ``override_img`` is ``None``.
    :param edge_thickness: The thickness of the edges of the adjacency graph.
    :param endpoint_radius: The radius of the nodes of the adjacency graph.
    :param endpoint_edge_thickness: The thickness of the border drawn around the nodes of the adjacency graph.
    :param edge_color: The color of the edges of the adjacency graph (RGB).
    :param endpoint_color: The color of the nodes of the adjacency graph (RGB).
    :param endpoint_edge_color: The color of the border drawn around the nodes of the adjacency graph (RGB).
    :param override_img: The image on top of which the adjacency graph is to be rendered. If ``None``, the (contrast-enhanced) raw image itensities will be used.
    :return: An object of type ``numpy.ndarray`` corresponding to an RGB image of the adjacency graph.

    .. hlist::
       :columns: 2

       - .. figure:: bbbc033-z28.png
            :width: 100%

            Original image (BBBC033).

       - .. figure:: ../../tests/expected/render.render_adjacencies/bbbc033-z28.png
            :width: 100%

            Result of using :py:meth:`~.render_adjacencies`.
    """
    if override_img is not None:
        assert override_img.ndim == 3 and override_img.shape[2] >= 3
        img = override_img[:, :, :3].copy()
        if (img > 1).any(): img = img / 255
    else:
        img = np.dstack([_fetch_image_from_data(data, normalize_img)] * 3)
        img = img / img.max()
    lines = data['adjacencies'].get_edge_lines()
    shape = img.shape[:2]
    for endpoint in data['seeds']:
        perim_mask  = skimage.draw.disk(endpoint, endpoint_radius + endpoint_edge_thickness, shape=shape)
        for i in range(3):
            img[:,:,i][ perim_mask] = endpoint_edge_color[i]
    for line in lines:
        line_buf  = draw_line(line[0], line[1], edge_thickness, shape=shape)
        line_mask = (line_buf > 0)
        line_vals = line_buf[line_mask]
        for i in range(3): img[:, :, i][line_mask] = (line_vals) * edge_color[i]
    for endpoint in data['seeds']:
        circle_mask = skimage.draw.disk(endpoint, endpoint_radius, shape=shape)
        for i in range(3):
            img[:,:,i][circle_mask] = endpoint_color[i]
    return (255 * img).clip(0, 255).astype('uint8')


def render_ymap(data, clim=None, cmap='bwr'):
    """Returns a visualization of the offset image intensities :math:`Y_\omega|_{\omega = \Omega}` (see :py:ref:`pipeline_theory_cvxprog`).

    :param data: The pipeline data object.
    :param clim: Tuple of the structure ``(cmin, cmax)``, where ``cmin`` and ``cmax`` are used for intensity clipping. The limits ``cmin`` and ``cmax`` are chosen automatically if ``clim`` is set to ``None``.
    :param cmap: Name of the color map to use for encoding the offset image intensities (see `the list <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_).
    :return: An object of type ``numpy.ndarray`` corresponding to an RGB image of the offset image intensities.

    .. hlist::
       :columns: 2

       - .. figure:: bbbc033-z28.png
            :width: 100%

            Original image (BBBC033).

       - .. figure:: ../../tests/expected/render.render_ymap/bbbc033-z28.png
            :width: 100%

            Result of using :py:meth:`~.render_ymap`.
    """
    y = data['y'] if isinstance(data, dict) else data
    if clim is None: clim = (-y.std(), +y.std())
    z = np.full((1, y.shape[1]), clim[0])
    z[0, -1] = clim[1]
    y = np.concatenate((z, y), axis=0)
    if isinstance(cmap, str): cmap = plt.cm.get_cmap(cmap)
    y  = y.clip(*clim)
    y -= y.min()
    y /= y.max()
    ymap = cmap(y)[1:]
    if ymap.ndim == 3 and ymap.shape[2] == 4: ymap = ymap[:,:,:3]
    return ymap


def normalize_image(img, spread=1, ret_minmax=False):
    """Performs contrast enhancement of an image.

    :param img: The image to be enhanced (object of ``numpy.ndarray`` type).
    :param spread: Governs the amount of enhancement. The lower the value, the stronger the enhancement.
    :param ret_minmax: ``True`` if the clipped image intensities should be returned and ``False`` otherwise.
    :return: The contrast-enhanced image if ``ret_minmax`` is ``False``, and a tuple of the structure ``(img, minval, maxval)`` if ``ret_minmax`` is ``True``, where ``img`` is the contrast-enhanced image, and ``minval`` and ``maxval`` are the clipped image intensities, respectively.

    .. hlist::
       :columns: 2

       - .. figure:: bbbc033-z28.png
            :width: 100%

            Original image (BBBC033).

       - .. figure:: ../../tests/expected/render.normalize_image/bbbc033-z28.png
            :width: 100%

            Result of using :py:meth:`~.normalize_image`.
    """
    if not np.allclose(img.std(), 0):
        minval, maxval = max([img.min(), img.mean() - spread * img.std()]), min([img.max(), img.mean() + spread * img.std()])
        img = img.clip(minval, maxval)
    else:
        minval, maxval = 0, 1
    img  = img - img.min()
    img /= img.max()
    return (img, minval, maxval) if ret_minmax else img


def _fetch_image_from_data(data, normalize_img=True):
    img = data['g_raw']
    if normalize_img: img = normalize_image(img)
    return img


def _fetch_rgb_image_from_data(data, normalize_img=True, override_img=None):
    if override_img is not None:
        img = override_img if override_img.ndim == 3 else np.dstack([override_img] * 3)
    elif 'g_rgb' in data:
        img = data['g_rgb']
        if img.max() > 1: img = img / 255
    else:
        img = data['g_raw']
        if normalize_img: img = normalize_image(img)
        img = np.dstack([img] * 3)
    img = img.copy()
    img[img < 0] = 0
    img[img > 1] = 1
    return img


def render_atoms(data, normalize_img=True, discarded_color=(0.3, 1, 0.3, 0.1), border_radius=2, border_color=(0,1,0), override_img=None):
    """Returns a visualization of the atomic image regions (see :ref:`pipeline_theory_c2freganal`).

    :param data: The pipeline data object.
    :param normalize_img: ``True`` if contrast-enhancement should be performed and ``False`` otherwise. Only used if ``override_img`` is ``None``.
    :param discarded_color: The color of image regions which are entirely discarded from processing (RGBA).
    :param border_radius: The half width of the borders of the atomic image regions.
    :param border_color: The color of the borders of the atomic image regions (RGB).
    :param override_img: The image on top of which the borders of the atomic image regions are to be rendered. If ``None``, the (contrast-enhanced) raw image itensities will be used.
    :return: An object of type ``numpy.ndarray`` corresponding to an RGB image of the atomic image regions.

    .. hlist::
       :columns: 2

       - .. figure:: bbbc033-z28.png
            :width: 100%

            Original image (BBBC033).

       - .. figure:: ../../tests/expected/render.render_atoms/bbbc033-z28.png
            :width: 100%

            Result of using :py:meth:`~.render_atoms`.
    """
    img = _fetch_image_from_data(data, normalize_img) if override_img is None else override_img
    return render_regions_over_image(img / img.max(), data['atoms'], background_label=0, bg=discarded_color, radius=border_radius, color=border_color)


def render_foreground_clusters(data, normalize_img=True, discarded_color=(0.3, 1, 0.3, 0.1), border_radius=2, border_color=(0,1,0), override_img=None):
    """Returns a visualization of regions of possibly clustered objects (see :ref:`pipeline_theory_jointsegandclustersplit`).

    :param data: The pipeline data object.
    :param normalize_img: ``True`` if contrast-enhancement should be performed and ``False`` otherwise. Only used if ``override_img`` is ``None``.
    :param discarded_color: The color of image regions which are entirely discarded from processing (RGBA).
    :param border_radius: The half width of the borders of the regions of possibly clustered objects.
    :param border_color: The color of the borders of the regions of possibly clustered objects (RGB).
    :param override_img: The image on top of which the borders of the regions of possibly clustered objects are to be rendered. If ``None``, the (contrast-enhanced) raw image itensities will be used.
    :return: An object of type ``numpy.ndarray`` corresponding to an RGB image of the regions of possibly clustered objects.

    .. hlist::
       :columns: 2

       - .. figure:: bbbc033-z28.png
            :width: 100%

            Original image (BBBC033).

       - .. figure:: ../../tests/expected/render.render_foreground_clusters/bbbc033-z28.png
            :width: 100%

            Result of using :py:meth:`~.render_foreground_clusters`.
    """
    img = _fetch_image_from_data(data, normalize_img) if override_img is None else override_img
    return render_regions_over_image(img / img.max(), data['clusters'], background_label=0, bg=discarded_color, radius=border_radius, color=border_color)


def rasterize_regions(regions, background_label=None, radius=3):
    """Returns the binary masks corresponding to the border of image regions and, optionally, the image background.

    :param regions: Integer-valued image (object of ``numpy.ndarray`` type) corresponding to the labels of different image regions.
    :param background_label: A designated label value, which is to be treated as the ``image background``.
    :param radius: The half width of the borders.
    :return: Tuple of the structure ``(borders, background)``, where ``borders`` is a binary mask of the borders of the image regions, and ``background`` is a binary mask of the union of those regions corresponding to the ``background_label``, if this is not ``None``. Otherwise, ``background`` is an binary mask filled with ``False`` values.
    """
    borders = np.zeros(regions.shape, bool)
    background = np.zeros(regions.shape, bool)
    for i in range(regions.max() + 1):
        region_mask = (regions == i)
        interior = morphology.erosion(region_mask, morphology.disk(radius))
        border   = np.logical_and(region_mask, ~interior)
        borders[border] = True
        if i == background_label: background = interior.astype(bool)
    return borders, background


def render_regions_over_image(img, regions, background_label=None, color=(0,1,0), bg=(0.6, 1, 0.6, 0.3), **kwargs):
    """Returns a visualization of image regions.

    :param img: The image on top of which the image regions are to be rendered.
    :param regions: Integer-valued image (object of ``numpy.ndarray`` type) corresponding to the labels of different image regions.
    :param background_label: A designated label value, which is to be treated as the image background.
    :param color: The color of the borders of the image regions (RGB).
    :param bg: The color of the image regions corresponding to the image background (RGBA). Only used if ``background_label`` is not ``None``.
    :param kwargs: Keyword arguments passed to the :py:meth:`~.rasterize_regions` function.
    :return: An object of type ``numpy.ndarray`` corresponding to an RGB image of the image regions.
    """
    assert img.ndim == 2 or (img.ndim == 3 and img.shape[2] in (1,3)), f'image has wrong dimensions: {img.shape}'
    if img.ndim == 2 or img.shape[2] == 1:
        result = np.zeros((img.shape[0], img.shape[1], 3))
        for i in range(3): result[:, :, i] = img
    else:
        result = img.copy()
    borders, background = rasterize_regions(regions, background_label, **kwargs)
    for i in range(3): result[:, :, i][borders] = color[i]
    for i in range(3): result[background, i] = bg[i] * bg[3] + result[background, i] * (1 - bg[3])
    return (255 * result).clip(0, 255).astype('uint8')


COLORMAP = {'r': [0], 'g': [1], 'b': [2], 'y': [0,1], 't': [1,2], 'w': [0,1,2]}


class ContourPaint:
    """Yields masks corresponding to contours of objects.

    :param fg_mask: Binary mask of the image foreground. Any contour never overlaps the image foreground except of those image regions corresponding to the contoured object itself.
    :param radius: The radius of the contour (half width).
    :param where: The position of the contour (``inner``, ``center``, or ``outer``).
    """

    def __init__(self, fg_mask, radius, where='center'):
        self.fg_mask = fg_mask
        self.where   = where
        self.radius  = radius
        self.selem   = morphology.disk(self.radius if where == 'center' else self.radius * 2)
        if where == 'outer':
            self.center_paint = ContourPaint(fg_mask, radius, where='center')
    
    def get_contour_mask(self, mask):
        """Returns the binary mask of the contour of an object.

        :param mask: Binary mask of an object.
        :return: Binary mask the contour
        """
        if self.where == 'center':
            contour = np.logical_xor(morphology.binary_erosion(mask, self.selem), morphology.binary_dilation(mask, self.selem))
        elif self.where == 'outer':
            contour = np.logical_xor(mask, morphology.binary_dilation(mask, self.selem))
            mask2   = np.logical_and(self.fg_mask, contour)
            contour = np.logical_and(contour, ~mask2)
            mask3   = morphology.binary_dilation(mask2, self.center_paint.selem)
            contour = np.logical_or(contour, np.logical_and(mask3, self.center_paint.get_contour_mask(mask)))
        elif self.where == 'inner':
            contour = np.logical_xor(mask, morphology.binary_erosion(mask, self.selem))
        return contour


def render_result_over_image(data, objects='postprocessed_objects', merge_overlap_threshold=np.inf, normalize_img=True, border_width=6, border_position='center', override_img=None, color='g'):
    """Returns a visualization of the segmentation result.

    By default, the segmentation result is rendered on top of the contrast-enhanced raw image intensities.

    :param data: The pipeline data object.
    :param objects: Either the name of the output which is to be treated as the segmentation result (see :ref:`pipeline_inputs_and_outputs`), or a list of :py:class:`~superdsm.objects.BaseObject` instances.
    :param merge_overlap_threshold: Any pair of two objects with an overlap larger than this threshold will be merged into a single object.
    :param normalize_img: ``True`` if contrast-enhancement should be performed and ``False`` otherwise. Only used if ``override_img`` is ``None``.
    :param border_width: The width of the contour to be drawn around the segmented objects.
    :param border_position: The position of the contour to be drawn around the segmented objects (``inner``, ``center``, or ``outer``).
    :param override_img: The image on top of which the contours of the segmented objects are to be rendered. If ``None``, the (contrast-enhanced) raw image itensities will be used.
    :param color: The color of the contour to be drawn around the segmented objects (``r`` for red, ``g`` for green, ``b`` for blue, ``y`` for yellow, ``t`` for teal, or ``w`` for white).
    :return: An object of type ``numpy.ndarray`` corresponding to an RGB image of the segmentation result.

    .. hlist::
       :columns: 2

       - .. figure:: bbbc033-z28.png
            :width: 100%

            Original image (BBBC033).

       - .. figure:: ../../tests/expected/render.render_result_over_image/bbbc033-z28.png
            :width: 100%

            Result of using :py:meth:`~.render_result_over_image`.
    """
    assert border_width % 2 == 0
    assert color in COLORMAP.keys()

    im_seg  = _fetch_rgb_image_from_data(data, normalize_img, override_img)
    im_seg /= im_seg.max()
    seg_objects = rasterize_labels(data, objects, merge_overlap_threshold=merge_overlap_threshold)
    cp = ContourPaint(seg_objects > 0, radius=border_width // 2, where=border_position)
    for l in set(seg_objects.flatten()) - {0}:
        seg_bnd = cp.get_contour_mask(seg_objects == l)
        colorchannels = COLORMAP[color]
        for i in range(3): im_seg[seg_bnd, i] = (1 if i in colorchannels else 0)
    return (255 * im_seg).round().clip(0, 255).astype('uint8')


def rasterize_objects(data, objects, dilate=0):
    """Generator which yields the segmentation masks of objects.

    :param data: The pipeline data object.
    :param objects: Either the name of the output which is to be rasterized (see :ref:`pipeline_inputs_and_outputs`), or a list of :py:class:`~superdsm.objects.BaseObject` instances.
    :param dilate: Dilates the segmentation mask of each object by this value, or erodes if negative.
    """
    if isinstance(objects, str): objects = [c for c in data[objects]]

    if dilate == 0:
        dlation, erosion = None, None
    else:
        dilation, erosion = (morphology.binary_dilation, morphology.binary_erosion)

    for foreground in render_objects_foregrounds(data['g_raw'].shape, objects):
        if dilate > 0:   foreground = dilation(foreground, morphology.disk( dilate))
        elif dilate < 0: foreground =  erosion(foreground, morphology.disk(-dilate))
        if foreground.any(): yield foreground.copy()


def rasterize_labels(data, objects='postprocessed_objects', merge_overlap_threshold=np.inf, dilate=0, background_label=0):
    """Returns an integer-valued image corresponding to uniquely labeled segmentation masks.

    :param data: The pipeline data object.
    :param objects: Either the name of the output which is to be rasterized (see :ref:`pipeline_inputs_and_outputs`), or a list of :py:class:`~superdsm.objects.BaseObject` instances.
    :param merge_overlap_threshold: Any pair of two objects with an overlap larger than this threshold will be merged into a single object.
    :param dilate: Dilates the segmentation mask of each object by this value, or erodes if negative.
    :param background_label: The label which is to be assigned to the image background. Must be non-positive.
    :return: An object of type ``numpy.ndarray`` corresponding the uniquely labeled segmentation masks.
    """
    assert background_label <= 0
    objects = [obj for obj in rasterize_objects(data, objects, dilate)]

    # First, we determine which objects overlap sufficiently
    merge_list = []
    merge_mask = [False] * len(objects)
    if merge_overlap_threshold <= 1:
        for i1, i2 in ((i1, i2) for i1, obj1 in enumerate(objects) for i2, obj2 in enumerate(objects[:i1])):
            obj1, obj2 = objects[i1], objects[i2]
            overlap = np.logical_and(obj1, obj2).sum() / (0. + min([obj1.sum(), obj2.sum()]))
            if overlap > merge_overlap_threshold:
                merge_list.append((i1, i2))  # i2 is always smaller than i1
                merge_mask[i1] = True

    # Next, we associate a (potentially non-unique) label to each object
    labels, obj_indices_by_label = list(range(1, 1 + len(objects))), {}
    for label, obj_idx in zip(labels, range(len(objects))): obj_indices_by_label[label] = [obj_idx]
    for merge_idx, merge_data in enumerate(merge_list):
        assert merge_data[1] < merge_data[0], 'inconsistent merge data'
        merge_label0  = len(objects) + 1 + merge_idx         # the new label for the merged objects
        merge_labels  = [labels[idx] for idx in merge_data]  # two labels of the objects to be merged
        if merge_labels[0] == merge_labels[1]: continue      # this can occur due to transitivity
        merge_indices = obj_indices_by_label[merge_labels[0]] + obj_indices_by_label[merge_labels[1]]
        for obj_idx in merge_indices: labels[obj_idx] = merge_label0
        obj_indices_by_label[merge_label0] = merge_indices
        for label in merge_labels: del obj_indices_by_label[label]
    del labels, merge_list, merge_mask

    # Finally, we merge the rasterized objects
    objects_by_label = dict((i[0], [objects[k] for k in i[1]]) for i in obj_indices_by_label.items())
    objects  = [(np.sum(same_label_objects, axis=0) > 0) for same_label_objects in objects_by_label.values()]
    result   = np.zeros(data['g_raw'].shape, 'uint16')
    if len(objects) > 0:
        overlaps = (np.sum(objects, axis=0) > 1)
        for l, obj in enumerate(objects, 1): result[obj] = l
        background = (result == 0).copy()
        result[overlaps] = 0
        dist = ndimage.morphology.distance_transform_edt(result == 0)
        result = segmentation.watershed(dist, result, mask=np.logical_not(background))

    # Work-around for this bug: https://github.com/scikit-image/scikit-image/issues/6587
    if result.dtype == np.int32:
        assert not (result < 0).any()
        assert not (result >= 2 ** 16).any()
        result = result.astype('uint16')

    # In rare cases it can happen that two or more objects overlap exactly, in which case the above code
    # will eliminate both objects. We will fix this by checking for such occasions explicitly:
    for obj in objects:
        obj_mask = ((result > 0) * 1 - (obj > 0) * 1 < 0)
        if obj_mask.any(): result[obj_mask] = result.max() + 1

    result[result == 0] = background_label
    return result


def shuffle_labels(labels, bg_label=None, seed=None):
    """Randomly shuffles the values of an integer-valued image.

    :param labels: An object of type ``numpy.ndarray`` corresponding to labeled segmentation masks.
    :param bg_label: If not ``None``, then this label stays fixed.
    :param seed: The seed used for randomization.
    :return: An object of type ``numpy.ndarray`` corresponding to ``labels`` with shuffled values (labels).
    """
    label_values0 = frozenset(labels.flatten())
    if bg_label is not None: label_values0 -= {bg_label}
    label_values0 = list(label_values0)
    if seed is not None: np.random.seed(seed)
    label_values1 = np.asarray(label_values0).copy()
    np.random.shuffle(label_values1)
    label_map = dict(zip(label_values0, label_values1))
    result = np.zeros_like(labels)
    for l in label_map.keys():
        cc = (labels == l)
        result[cc] = label_map[l]
    return result


def colorize_labels(labels, bg_label=0, cmap='gist_rainbow', bg_color=(0,0,0), shuffle=None):
    """Returns a colorized representation of an integer-valued image.

    :param labels: An object of type ``numpy.ndarray`` corresponding to labeled segmentation masks.
    :param bg_label: Image areas with this label are forced to the color ``bg_color``.
    :param cmap: The colormap used to colorize the remaining labels (see `the list <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_).
    :param bg_color: The color used to represent the image regions with label ``bg_label`` (RGB).
    :param shuffle: If not ``None``, then used as ``seed`` to shuffle the labels before colorization (see :py:meth:`~.shuffle_labels`), and not used otherwise.
    :return: An object of type ``numpy.ndarray`` corresponding to an RGB image.

    .. hlist::
       :columns: 2

       - .. figure:: bbbc033-z28.png
            :width: 100%

            Original image (BBBC033).

       - .. figure:: ../../tests/expected/render.colorize_labels/bbbc033-z28.png
            :width: 100%

            Result of using :py:meth:`~.rasterize_labels` and :py:meth:`~.colorize_labels`.
    """
    if shuffle is not None:
        labels = shuffle_labels(labels, bg_label=bg_label, seed=shuffle)
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    img = cmap((labels - labels.min()) / float(labels.max() - labels.min()))
    if img.shape[2] > 3: img = img[:,:,:3]
    if bg_label is not None:
        bg = (labels == bg_label)
        img[bg] = np.asarray(bg_color)[None, None, :]
    return img

