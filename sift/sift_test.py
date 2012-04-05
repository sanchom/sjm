import itertools
import glob
import math
import unittest

import sift_util
import sift_descriptors_pb2
import tempfile

class TestSiftUtilFunctions(unittest.TestCase):
    def test_read_parameters(self):
        extraction_parameters = \
            sift_util.get_extraction_parameters('test_data/seminar.sift')
        self.assertEqual(extraction_parameters.normalization_threshold, 0.5)
        self.assertEqual(extraction_parameters.rotation_invariance, False)
        self.assertEqual(extraction_parameters.discard_unnormalized, True)
        self.assertEqual(extraction_parameters.multiscale, True)
        self.assertEqual(extraction_parameters.percentage, 1.0)
        self.assertEqual(extraction_parameters.minimum_radius, 16)
        self.assertEqual(extraction_parameters.fractional_xy, True)
        self.assertEqual(extraction_parameters.resolution_factor, 1)
        self.assertEqual(extraction_parameters.top_left_x, 0)
        self.assertEqual(extraction_parameters.top_left_y, 0)
        self.assertEqual(extraction_parameters.bottom_right_x, 4294967295)
        self.assertEqual(extraction_parameters.bottom_right_y, 4294967295)
        self.assertAlmostEqual(extraction_parameters.first_level_smoothing, 1.3)

    def test_read_descriptor(self):
        descriptors = sift_util.load_descriptors('test_data/seminar.sift')
        self.assertEqual(1369, len(descriptors.sift_descriptor))

    def test_count(self):
        self.assertEqual(sift_util.count_descriptors_in_list([]), 0)
        self.assertEqual(sift_util.count_descriptors_in_file(
                'test_data/seminar.sift'), 1369)
        self.assertEqual(sift_util.count_descriptors_in_list(
                ['test_data/seminar.sift', 'test_data/seminar.sift']), 1369 * 2)

    def test_read_numpy_array_from_files_noalpha(self):
        """ Check the reading of sift descriptors from files.

        This doesn't check the application of the alpha parameter, just
        number of descriptors loaded and that max_points is being
        observed.
        """
        # Test loading from empty list
        point_array = sift_util.load_array_from_files(file_list=[],
                                                      max_points=300)
        self.assertEqual(len(point_array), 0)
        # Test loading with max points less than num available
        point_array = sift_util.load_array_from_files(
            file_list=glob.glob('test_data/seminar.sift'), max_points=300)
        self.assertTrue(len(point_array) > 0)
        self.assertTrue(len(point_array) <= 300)
        # Test loading with max points equal to num available
        point_array = sift_util.load_array_from_files(
            file_list=glob.glob('test_data/seminar.sift'), max_points=1369)
        self.assertEqual(len(point_array), 1369)
        # Test loading with max points not specified
        point_array = sift_util.load_array_from_files(
            file_list=glob.glob('test_data/seminar.sift'))
        self.assertEqual(len(point_array), 1369)
        # Test loading with max points higher than num available
        point_array = sift_util.load_array_from_files(
            file_list=glob.glob('test_data/seminar.sift'), max_points=1500)
        self.assertEqual(len(point_array), 1369)
        # Test loading from list with max points lower than available
        point_array = sift_util.load_array_from_files(
            file_list=glob.glob('test_data/*.sift'), max_points=1369)
        self.assertTrue(len(point_array) <= 1369)
        # Test loading from list with max points lower than available, but higher than a single file
        point_array = sift_util.load_array_from_files(
            file_list=glob.glob('test_data/*.sift'), max_points=1500)
        self.assertTrue(len(point_array) > 1369) # Check that more descriptors than from a single file
        self.assertTrue(len(point_array) <= 1500) # Check that within max_points

    def test_read_numpy_array_files_alpha(self):
        """ Checks the use of the alpha parameter in the loading of sift files.
        """
        # Test loading with alpha = 0
        point_array = sift_util.load_array_from_files(file_list=glob.glob('test_data/seminar.sift'), max_points=1369)
        self.assertEqual(point_array.shape[1], 128)
        point_array_1_0 = sift_util.load_array_from_files(file_list=glob.glob('test_data/seminar.sift'), alpha=1.0, max_points=1369)
        self.assertEqual(point_array_1_0.shape[1], 130)
        point_array_0_1 = sift_util.load_array_from_files(file_list=glob.glob('test_data/seminar.sift'), alpha=0.1, max_points=1369)
        self.assertEqual(point_array_0_1.shape[1], 130)

        for d_1, d_2 in itertools.izip(point_array_1_0, point_array_0_1):
            self.assertTrue(math.fabs(d_1[-2] * 0.1 - d_2[-2]) <= 1)
            self.assertTrue(math.fabs(d_1[-1] * 0.1 - d_2[-1]) <= 1)

    def test_read_numpy_array_files_alpha_over_one(self):
        """ Checks the use of the alpha parameter in the loading of sift files.
        """
        # Test loading with alpha = 0
        point_array = sift_util.load_array_from_files(file_list=glob.glob('test_data/seminar.sift'), max_points=1369)
        self.assertEqual(point_array.shape[1], 128)
        point_array_1_0 = sift_util.load_array_from_files(file_list=glob.glob('test_data/seminar.sift'), alpha=1.0, max_points=1369)
        self.assertEqual(point_array_1_0.shape[1], 130)
        point_array_2_0 = sift_util.load_array_from_files(file_list=glob.glob('test_data/seminar.sift'), alpha=2.0, max_points=1369)
        self.assertEqual(point_array_2_0.shape[1], 130)

        for d_1, d_2 in itertools.izip(point_array_1_0, point_array_2_0):
            self.assertTrue(math.fabs(d_1[-2] * 2.0 - d_2[-2]) <= 1, 'd_1[-2] = %d, d_2[-2] = %d' % (d_1[-2], d_2[-2]))
            self.assertTrue(math.fabs(d_1[-1] * 2.0 - d_2[-1]) <= 1, 'd_1[-1] = %d, d_2[-1] = %d' % (d_1[-1], d_2[-1]))

    def test_protobuf_to_numpy_converstion(self):
        descriptor = sift_descriptors_pb2.SiftDescriptor()
        descriptor.bin.append(15)
        descriptor.bin.append(35)
        descriptor.x = 0.2
        descriptor.y = 0.3
        descriptor.scale = 1
        descriptor_array = sift_util.convert_protobuf_descriptor_to_weighted_array(descriptor)
        self.assertEqual(len(descriptor_array), 2)
        self.assertAlmostEqual(descriptor_array[0], 15)
        self.assertAlmostEqual(descriptor_array[1], 35)
        alpha = 0.5
        descriptor_array = sift_util.convert_protobuf_descriptor_to_weighted_array(descriptor, alpha)
        self.assertEqual(len(descriptor_array), 4)
        self.assertAlmostEqual(descriptor_array[0], 15)
        self.assertAlmostEqual(descriptor_array[1], 35)
        self.assertAlmostEqual(descriptor_array[2], int(0.2 * 127 * alpha + 0.5))
        self.assertAlmostEqual(descriptor_array[3], int(0.3 * 127 * alpha + 0.5))
        alpha = 1.5
        descriptor_array = sift_util.convert_protobuf_descriptor_to_weighted_array(descriptor, alpha)
        self.assertEqual(len(descriptor_array), 4)
        self.assertAlmostEqual(descriptor_array[0], 15)
        self.assertAlmostEqual(descriptor_array[1], 35)
        self.assertAlmostEqual(descriptor_array[2], int(0.2 * 127 * alpha + 0.5))
        self.assertAlmostEqual(descriptor_array[3], int(0.3 * 127 * alpha + 0.5))

        
    def test_merge_descriptor_sets(self):
        files = ['test_data/seminar.sift', 'test_data/Glass_is_Liquide.sift']

        descriptor_set_list = []
        for f in files:
            descriptor_set = sift_util.load_descriptors(f)
            descriptor_set_list.append(descriptor_set)

        merged_set = sift_util.merge_descriptor_sets(descriptor_set_list)
        self.assertEqual(sift_util.count_descriptors_in_list(files),
                         len(merged_set.sift_descriptor))

    def test_convert_bare_to_params_prepend(self):
        """ Tests conversion from a bare protobuf to one with params prepended.
        """
        destination = tempfile.TemporaryFile()
        original = open('test_data/seminar.old-sift', 'rb')
        sift_util.convert_bare_set_to_set_with_params(original, destination)
        self.assertEqual(original.read(), destination.read())

if __name__ == '__main__':
    unittest.main()
