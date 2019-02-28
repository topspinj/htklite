import numpy as np
import os
import skimage.io
import unittest

from htklite.preprocessing import color_deconvolution, color_convolution

TEST_DATA_DIR = 'tests/sample_images/'

class ColorDeconvolutionTest(unittest.TestCase):

    def test_roundtrip(self):
        im_path = os.path.join(TEST_DATA_DIR, 'Easy1.png')
        im = skimage.io.imread(im_path)[..., :3]

        w = np.array([[0.650, 0.072, 0],
                      [0.704, 0.990, 0],
                      [0.286, 0.105, 0]])

        conv_result = color_deconvolution(im, w, 255)

        im_reconv = color_convolution(conv_result.StainsFloat,
                                              conv_result.Wc, 255)

        np.testing.assert_allclose(im, im_reconv, atol=1)