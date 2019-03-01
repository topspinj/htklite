import numpy as np
from htklite.preprocessing import color_conversion
from htklite.utils import convert_image_to_matrix, convert_matrix_to_image

def color_convolution(im_stains, w, I_0=None):
    """Performs Color Convolution
    Reconstructs a color image from the stain matrix `w` and
    the individual images stored as channels in `im_stains` and generated
    by ColorDeconvolution.

    Parameters
    ----------
    im_stains : array_like
        An RGB image where in each channel contains image of one stain
    w : array_like
        A 3x3 matrix containing the stain colors in its columns.
        In the case of two stains, the third column is zero and will be
        complemented using cross-product. The matrix should contain a
        minumum two nonzero columns.
    I_0 : float or array_like, optional
        A float a 3-vector containing background RGB intensities.
        If unspecified, use the old OD conversion.

    Returns
    -------
    im_rgb : array_like
        Reconstructed RGB image with intensity values ranging from [0, 255],
        suitable for display.
    """

    # transform 3D input stain image to 2D stain matrix format
    m = convert_image_to_matrix(im_stains)

    # transform input stains to optical density values, convolve and
    # tfm back to stain
    sda_fwd = color_conversion.rgb_to_sda(m, 255 if I_0 is not None else None,
                                          allow_negatives=True)
    sda_conv = np.dot(w, sda_fwd)
    sda_inv = color_conversion.sda_to_rgb(sda_conv, I_0)

    # reshape output, transform type
    im_rgb = (convert_matrix_to_image(sda_inv, im_stains.shape)
              .clip(0, 255).astype(np.uint8))

    return im_rgb
