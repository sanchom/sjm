# Copyright 2010 Sancho McCann
# Author: Sancho McCann

import glob
import os
import random
import unittest

from naive_bayes_nearest_neighbor import caltech_util
from sift import sift_descriptors_pb2
from sift import sift_util

class TestCaltechHelperFunctions(unittest.TestCase):
    def tearDown(self):
        if os.path.exists('/tmp/seminar.sift'):
            os.remove('/tmp/seminar.sift')

    def test_do_extraction_produces_output(self):
        """ Tests working case do_extraction """
        image_path = os.path.abspath('../test_images/seminar.pgm')
        destination_dir = '/tmp/'
        parameters = sift_descriptors_pb2.ExtractionParameters()
        caltech_util.do_extraction((image_path, destination_dir,
                                    parameters.SerializeToString()))
        expected_output_path = os.path.join(destination_dir, 'seminar.sift')
        self.assertTrue(os.path.exists(expected_output_path))
        self.assertTrue(os.path.getsize(expected_output_path) > 0)

    def test_do_extraction_doesnt_extract_when_already_done(self):
        image_path = os.path.abspath('../test_images/seminar.pgm')
        destination_dir = '/tmp/'
        expected_output_path = os.path.join(destination_dir, 'seminar.sift')

        parameters = sift_descriptors_pb2.ExtractionParameters()
        caltech_util.do_extraction((image_path, destination_dir,
                                    parameters.SerializeToString()))
        first_creation_time_check = os.path.getmtime(expected_output_path)
        caltech_util.do_extraction((image_path, destination_dir,
                                    parameters.SerializeToString()))
        second_creation_time_check = os.path.getmtime(expected_output_path)
        self.assertEqual(first_creation_time_check, second_creation_time_check)

    def test_do_extraction_does_extract_when_params_dont_match_existing(self):
        image_path = os.path.abspath('../test_images/seminar.pgm')
        destination_dir = '/tmp/'
        expected_output_path = os.path.join(destination_dir, 'seminar.sift')

        parameters = sift_descriptors_pb2.ExtractionParameters()
        caltech_util.do_extraction((image_path, destination_dir,
                                    parameters.SerializeToString()))
        first_creation_time_check = os.path.getmtime(expected_output_path)
        parameters.multiscale = False
        caltech_util.do_extraction((image_path, destination_dir,
                                    parameters.SerializeToString()))
        second_creation_time_check = os.path.getmtime(expected_output_path)
        self.assertNotEqual(first_creation_time_check,
                            second_creation_time_check)

    def test_extraction_output_matches_requested_default_parameters(self):
        """ Tests that the extraction has been performed with the requested
        parameters.
        """
        image_path = os.path.abspath('../test_images/seminar.pgm')
        destination_dir = '/tmp'

        # Tests the default settings.
        parameters = sift_descriptors_pb2.ExtractionParameters()
        caltech_util.do_extraction((image_path, destination_dir,
                                    parameters.SerializeToString()))
        expected_output_path = os.path.join(destination_dir, 'seminar.sift')
        result_params = \
            sift_util.get_extraction_parameters(expected_output_path)
        self.assertEqual(result_params.rotation_invariance, False)
        self.assertEqual(result_params.normalization_threshold, 0)
        self.assertEqual(result_params.discard_unnormalized, False)
        self.assertEqual(result_params.multiscale, True)
        self.assertEqual(result_params.percentage, 1)
        self.assertEqual(result_params.minimum_radius, 0)
        self.assertEqual(result_params.fractional_xy, False)
        self.assertEqual(result_params.top_left_x, 0)
        self.assertEqual(result_params.top_left_y, 0)
        self.assertEqual(result_params.bottom_right_x, 2147483647)
        self.assertEqual(result_params.bottom_right_y, 2147483647)
        self.assertEqual(result_params.implementation,
                         sift_descriptors_pb2.ExtractionParameters.VLFEAT)
        self.assertEqual(result_params.grid_method,
                         sift_descriptors_pb2.ExtractionParameters.FIXED_3X3)
        self.assertEqual(result_params.smoothed, True)
        self.assertAlmostEqual(result_params.first_level_smoothing, 0.6666666,
                               places=4)

    def test_extraction_output_matches_requested_parameters_nondefault(self):
        """ Tests that the extraction has been performed with the requested
        parameters.
        """
        image_path = os.path.abspath('../test_images/seminar.pgm')
        destination_dir = '/tmp'

        # Tests the default settings + multisale = False
        parameters = sift_descriptors_pb2.ExtractionParameters()
        parameters.multiscale = False
        parameters.fractional_xy = True
        parameters.percentage = 0.5
        parameters.normalization_threshold = 0.5
        parameters.minimum_radius = 16
        parameters.discard_unnormalized = True
        parameters.first_level_smoothing = 0.5
        parameters.grid_method = \
            sift_descriptors_pb2.ExtractionParameters.SCALED_BIN_WIDTH
        caltech_util.do_extraction((image_path, destination_dir,
                                    parameters.SerializeToString()))
        expected_output_path = os.path.join(destination_dir, 'seminar.sift')
        result_params = \
            sift_util.get_extraction_parameters(expected_output_path)
        self.assertEqual(result_params.rotation_invariance, False)
        self.assertEqual(result_params.normalization_threshold, 0.5)
        self.assertEqual(result_params.discard_unnormalized, True)
        self.assertEqual(result_params.multiscale, False)
        self.assertEqual(result_params.percentage, 0.5)
        self.assertEqual(result_params.minimum_radius, 16)
        self.assertEqual(result_params.fractional_xy, True)
        self.assertEqual(result_params.top_left_x, 0)
        self.assertEqual(result_params.top_left_y, 0)
        self.assertEqual(result_params.bottom_right_x, 2147483647)
        self.assertEqual(result_params.bottom_right_y, 2147483647)
        self.assertEqual(result_params.grid_method,
                         sift_descriptors_pb2.ExtractionParameters.SCALED_BIN_WIDTH)
        self.assertEqual(result_params.implementation,
                         sift_descriptors_pb2.ExtractionParameters.VLFEAT)
        self.assertEqual(result_params.smoothed, True)
        self.assertAlmostEqual(result_params.first_level_smoothing, 0.5,
                               places=4)

    def test_do_extraction_no_image(self):
        """ Tests image missing do_extraction """
        image_path = os.path.abspath('../test_images/missing.pgm')
        destination_dir = '/tmp/'
        parameters = sift_descriptors_pb2.ExtractionParameters()
        self.assertRaises(caltech_util.ImageNotFoundError,
                          caltech_util.do_extraction,
                          (image_path, destination_dir,
                           parameters.SerializeToString()))
        expected_output_path = os.path.join(destination_dir, 'missing.sift')
        self.assertFalse(os.path.exists(expected_output_path))

if __name__ == '__main__':
    unittest.main()
