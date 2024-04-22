"""
Copyright 2017-2024 Leonid Kostrykin, Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
"""

import contextlib

import skimage.io


def imread(*args, **kwargs):
    """
    Wrapper around ``skimage.io.imread`` which mutes non-fatal errors.

    When using ``skimage.io.imread`` to read an image file, sometimes errors can be reported although the image file will be read successfully.
    In those cases, Galaxy might detect the errors and assume that the tool has failed: https://docs.galaxyproject.org/en/latest/dev/schema.html#error-detection
    To prevent this, this wrapper around ``skimage.io.imread`` will mute non-fatal all errors.
    """
    try:

        # Mute stderr unless an error occurs
        with contextlib.redirect_stderr(None):
            return skimage.io.imread(*args, **kwargs)

    except:  # noqa: E722

        # Raise the error outside of the contextlib redirection
        raise
