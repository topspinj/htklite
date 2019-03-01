from .sda_to_rgb import sda_to_rgb


def od_to_rgb(im_od):
    """Transforms input optical density image `im_od` into RGB space

    Parameters
    ----------
    im_od : array_like
        A floating-point image of optical density values obtained
        from rgb_to_od.

    Returns
    -------
    im_rgb : array_like
        A floating-point multi-channel image with intensity
        values in the range [0, 255].
    """

    return sda_to_rgb(im_od, None)  # compatibility mode
