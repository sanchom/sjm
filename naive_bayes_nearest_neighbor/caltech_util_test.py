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

    def test_extraction_list(self):
        """ Tests creation of extraction list for the caltech dataset. """
        caltech_data = '/lci/project/lowe/sanchom/101_ObjectCategories-128'
        caltech_local = '/var/tmp/sanchom/caltech_local'
        extraction_list = caltech_util.build_extraction_list(caltech_data, caltech_local)

        for (original_file, target_directory) in extraction_list:
            self.assertTrue(os.path.isfile(original_file))
            original_directory_relative = os.path.dirname(original_file).replace(caltech_data, "")
            target_directory_relative = target_directory.replace(caltech_local, "")
            self.assertEqual(original_directory_relative, target_directory_relative)

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

    def test_train_test_split(self):
        """ Tests the creation of training and testing lists.

        Checks proper prevention of overlap between folds, enforcement of 
        size requests, etc.
        """
        file_list = map(lambda x: str(x) + '.jpg', range(10))
        random.shuffle(file_list)
        
        # Single folds
        (training_list, testing_list) = \
            caltech_util.split_into_train_test(file_list, num_train=0, num_test=0)
        self.assertEqual(len(training_list), 0) # Expected length
        self.assertEqual(len(testing_list), 0) # Expected length

        (training_list, testing_list) = \
            caltech_util.split_into_train_test(file_list, num_train=1, num_test=1)
        self.assertEqual(len(training_list), 1) # Expected length
        self.assertEqual(len(testing_list), 1) # Expected length
        self.assertEqual(len(set(training_list) & set(testing_list)), 0) # No intersection

        (training_list, testing_list) = \
            caltech_util.split_into_train_test(file_list, num_train=2, num_test=3)
        self.assertEqual(len(training_list), 2) # Expected length
        self.assertEqual(len(testing_list), 3) # Expected length
        self.assertEqual(len(set(training_list) & set(testing_list)), 0) # No intersection

        # Multiple folds
        (training_list_fold_0, testing_list_fold_0) = \
            caltech_util.split_into_train_test(file_list, num_train=2, num_test=3, fold_id=0)
        self.assertEqual(len(training_list_fold_0), 2) # Expected length
        self.assertEqual(len(testing_list_fold_0), 3) # Expected length
        self.assertEqual(len(set(training_list_fold_0) & set(testing_list_fold_0)), 0) # No intersection

        (training_list_fold_1, testing_list_fold_1) = \
            caltech_util.split_into_train_test(file_list, num_train=2, num_test=3, fold_id=1)
        self.assertEqual(len(training_list_fold_1), 2) # Expected length
        self.assertEqual(len(testing_list_fold_1), 3) # Expected length
        # No intersection between training and test sets
        self.assertEqual(len(set(training_list_fold_1) & set(testing_list_fold_1)), 0)
        # No intersection between training folds
        self.assertEqual(len(set(training_list_fold_0) & set(training_list_fold_1)), 0)

        # Fold that checks wrap around condition
        (training_list_fold_3, testing_list_fold_3) = \
            caltech_util.split_into_train_test(file_list, num_train=2, num_test=3, fold_id=3)
        self.assertEqual(len(training_list_fold_3), 2) # Expected length
        self.assertEqual(len(testing_list_fold_3), 3) # Expected length
        # No intersection between training and test sets
        self.assertEqual(len(set(training_list_fold_3) & set(testing_list_fold_3)), 0)
        # No intersection between training folds
        self.assertEqual(len(set(training_list_fold_0) & set(training_list_fold_3)), 0)
        self.assertEqual(len(set(training_list_fold_1) & set(training_list_fold_3)), 0)

        # Max number of folds requested
        (training_list, testing_list) = \
            caltech_util.split_into_train_test(file_list, num_train=2, num_test=3, fold_id=4)
        self.assertEqual(len(training_list), 2) # Expected length
        self.assertEqual(len(testing_list), 3) # Expected length
        self.assertEqual(len(set(training_list) & set(testing_list)), 0) # No intersection

        # One past max number of folds requested
        self.assertRaises(caltech_util.FoldIdError,
                          caltech_util.split_into_train_test,
                          file_list, 2, 3, 5)

        # Too many training files requested
        self.assertRaises(caltech_util.TrainSizeError,
                          caltech_util.split_into_train_test,
                          file_list, 11, 3, 0)

        # Too many test files requested
        self.assertRaises(caltech_util.TestSizeError,
                          caltech_util.split_into_train_test,
                          file_list, 2, 9, 0)

if __name__ == '__main__':
    unittest.main()
