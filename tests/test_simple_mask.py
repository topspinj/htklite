import os
import unittest
import numpy as np
import skimage.io

from htklite.segmentation import simple_mask

TEST_DATA_DIR = 'tests/sample_images/'

class ColorDeconvolutionTest(unittest.TestCase):

    def test_simple_mask_output(self):
        im_path = os.path.join(TEST_DATA_DIR, 'Easy1.png')
        img = skimage.io.imread(im_path)[..., :3]
        output = simple_mask(img)
        self.assertIsInstance(output, np.ndarray)