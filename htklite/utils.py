def convert_image_to_matrix(im):
    """Convert an image (MxNx3 array) to a column matrix of pixels
    (3x(M*N)).  It will pass through a 2D array unchanged.

    """
    if im.ndim == 2:
        return im

    return im.reshape((-1, im.shape[-1])).T

def convert_matrix_to_image(m, shape):
    """Convert a column matrix of pixels to a 3D image given by shape.
    The number of channels is taken from m, not shape.  If shape has
    length 2, the matrix is returned unchanged.  This is the inverse
    of convert_image_to_matrix:

    im == convert_matrix_to_image(convert_image_to_matrix(im),
    im.shape)

    """
    if len(shape) == 2:
        return m

    return m.T.reshape(shape[:-1] + (m.shape[0],))
