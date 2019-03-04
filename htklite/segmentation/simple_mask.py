import numpy as np
from skimage import color
from sklearn.neighbors.kde import KernelDensity
from scipy.stats import norm
from scipy.optimize import fmin_slsqp
from scipy import signal


def simple_mask(im_rgb, bandwidth=2, bgnd_std=2.5, tissue_std=30,
                min_peak_width=10, max_peak_width=25,
                fraction=0.10, min_tissue_prob=0.05):
    """Performs segmentation of the foreground (tissue)
    Uses a simple two-component Gaussian mixture model to mask tissue areas
    from background in brightfield H&E images. Kernel-density estimation is
    used to create a smoothed image histogram, and then this histogram is
    analyzed to identify modes corresponding to tissue and background. The
    mode peaks are then analyzed to estimate their width, and a constrained
    optimization is performed to fit gaussians directly to the histogram
    (instead of using expectation-maximization directly on the data which
    is more prone to local minima effects). A maximum-likelihood threshold
    is then derived and used to mask the tissue area in a binarized image.

    Parameters
    ----------
    im_rgb : array_like
        An RGB image of type unsigned char.
    bandwidth : double, optional
        Bandwidth for kernel density estimation - used for smoothing the
        grayscale histogram. Default value = 2.
    bgnd_std : double, optional
        Standard deviation of background gaussian to be used if
        estimation fails. Default value = 2.5.
    tissue_std: double, optional
        Standard deviation of tissue gaussian to be used if estimation fails.
        Default value = 30.
    min_peak_width: double, optional
        Minimum peak width for finding peaks in kde histogram. Used to
        initialize curve fitting process. Default value = 10.
    max_peak_width: double, optional
        Maximum peak width for finding peaks in kde histogram. Used to
        initialize curve fitting process. Default value = 25.
    fraction: double, optional
        Fraction of pixels to sample for building foreground/background
        model. Default value = 0.10.
    min_tissue_prob : double, optional
        Minimum probability to qualify as tissue pixel. Default value = 0.05.

    Returns
    -------
    im_mask : array_like
        A binarized version of `I` where foreground (tissue) has value '1'.
    """

    # convert image to grayscale, flatten and sample
    im_rgb = 255 * color.rgb2gray(im_rgb)
    im_rgb = im_rgb.astype(np.uint8)
    num_samples = np.int(fraction * im_rgb.size)
    img_samples = np.random.choice(im_rgb.flatten(), num_samples)[:, np.newaxis]

    # kernel-density smoothed histogram
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(img_samples)
    x_hist = np.linspace(0, 255, 256)[:, np.newaxis]
    y_hist = np.exp(kde.score_samples(x_hist))[:, np.newaxis]
    y_hist = y_hist / sum(y_hist)

    # flip smoothed y-histogram so that background mode is on the left side
    y_hist = np.flipud(y_hist)

    # identify initial mean parameters for gaussian mixture distribution
    # take highest peak among remaining peaks as background
    peaks = signal.find_peaks_cwt(y_hist.flatten(),
                                  np.arange(min_peak_width, max_peak_width))
    bg_peak = peaks[0]
    if len(peaks) > 1:
        tissue_peak = peaks[y_hist[peaks[1:]].argmax() + 1]
    else:  # no peak found - take initial guess at 2/3 distance from origin
        tissue_peak = np.asscalar(x_hist[np.round(0.66*x_hist.size)])

    # analyze background peak to estimate variance parameter via FWHM
    bg_scale = estimate_variance(x_hist, y_hist, bg_peak)
    if bg_scale == -1:
        bg_scale = bgnd_std

    # analyze tissue peak to estimate variance parameter via FWHM
    tissue_scale = estimate_variance(x_hist, y_hist, tissue_peak)
    if tissue_scale == -1:
        tissue_scale = tissue_std

    # solve for mixing parameter
    mix = y_hist[bg_peak] * (bg_scale * (2 * np.pi)**0.5)

    # flatten kernel-smoothed histogram and corresponding x values for
    # optimization
    x_hist = x_hist.flatten()
    y_hist = y_hist.flatten()

    # define gaussian mixture model
    def gaussian_mixture(x, mu1, mu2, sigma1, sigma2, p):
        rv1 = norm(loc=mu1, scale=sigma1)
        rv2 = norm(loc=mu2, scale=sigma2)
        return p * rv1.pdf(x) + (1 - p) * rv2.pdf(x)

    # define gaussian mixture model residuals
    def gaussian_residuals(params, y, x):
        mu1, mu2, sigma1, sigma2, p = params
        yhat = gaussian_mixture(x, mu1, mu2, sigma1, sigma2, p)
        return sum((y - yhat) ** 2)

    # fit Gaussian mixture model and unpack results
    params = fmin_slsqp(gaussian_residuals,
                            [bg_peak, tissue_peak, bg_scale, tissue_scale, mix],
                            args=(y_hist, x_hist),
                            bounds=[(0, 255), (0, 255),
                                    (np.spacing(1), 10),
                                    (np.spacing(1), 50), (0, 1)],
                            iprint=0)

    mean_background = params[0]
    mean_tissue = params[1]
    sigma_background = params[2]
    sigma_tissue = params[3]
    p = params[4]

    # create mask based on Gaussian mixture model
    background = norm(loc=mean_background, scale=sigma_background)
    tissue = norm(loc=mean_tissue, scale=sigma_tissue)
    prob_background = p * background.pdf(x_hist)
    prob_tissue = (1 - p) * tissue.pdf(x_hist)

    # identify maximum likelihood threshold
    difference = prob_tissue - prob_background
    candidates = np.nonzero(difference >= 0)[0]
    Filtered = np.nonzero(x_hist[candidates] > mean_background)
    max_likelihood = x_hist[candidates[Filtered[0]][0]]

    # identify limits for tissue model (MinProb, 1-MinProb)
    endpoints = np.asarray(tissue.interval(1 - min_tissue_prob / 2))

    # invert threshold and tissue mean
    max_likelihood = 255 - max_likelihood
    mean_tissue = 255 - mean_tissue
    endpoints = np.sort(255 - endpoints)

    # generate mask
    im_mask = (im_rgb <= max_likelihood) & (im_rgb >= endpoints[0]) & \
              (im_rgb <= endpoints[1])
    im_mask = im_mask.astype(np.uint8)

    return im_mask


def estimate_variance(x, y, peak):
    """Estimates variance of a peak in a histogram using the full-width at half-maximum
    (FWHM) of an approximate normal distribution. Starting from a user-supplied 
    peak and histogram, this method traces down each side of the peak to estimate 
    the FWHM and variance of the peak. If tracing fails on either side, the FWHM is
    estimated as twice the half-width at half-maximum (HWHM).

    Parameters
    ----------
    x : array_like
        vector of x-histogram locations.
    y : array_like
        vector of y-histogram locations.
    peak : double
        index of peak in y at which variance is estimated
    
    Returns
    -------
    scale : double
        Standard deviation of normal distribution approximating peak. Value is
        -1 if fitting process fails.
    """

    # analyze peak to estimate variance parameter via FWHM
    left = peak
    while y[left] > y[peak] / 2 and left >= 0:
        left -= 1
        if left == -1:
            break
    right = peak
    while y[right] > y[peak] / 2 and right < y.size:
        right += 1
        if right == y.size:
            break
    if left != -1 and right != y.size:
        left_slope = y[left + 1] - y[left] / (x[left + 1] - x[left])
        left = (y[peak] / 2 - y[left]) / left_slope + x[left]
        right_slope = y[right] - y[right - 1] / (x[right] - x[right - 1])
        right = (y[peak] / 2 - y[right]) / right_slope + x[right]
        scale = (right - left) / 2.355
    if left == -1:
        if right == y.size:
            scale = -1
        else:
            right_slope = y[right] - y[right - 1] / (x[right] - x[right - 1])
            right = (y[peak] / 2 - y[right]) / right_slope + x[right]
            scale = 2 * (right - x[peak]) / 2.355
    if right == y.size:
        if left == -1:
            scale = -1
        else:
            left_slope = y[left + 1] - y[left] / (x[left + 1] - x[left])
            left = (y[peak] / 2 - y[left]) / left_slope + x[left]
            scale = 2 * (x[peak] - left) / 2.355
    return scale