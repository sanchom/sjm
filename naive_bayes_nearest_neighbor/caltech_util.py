# Copyright (c) 2010, Sancho McCann

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import glob
import itertools
import math
import multiprocessing
import os
import random
import subprocess
import sys

from sift import sift_descriptors_pb2
from sift import sift_util

class CaltechUtilError(Exception):
    """ Base class for errors in the caltech_util module """

class TestSizeError(CaltechUtilError):
    """ Raise when requesting too much testing data. """
    pass

class TrainSizeError(CaltechUtilError):
    """ Raise when requesting too much training data. """
    pass

class ImageNotFoundError(CaltechUtilError):
    """ Raise when an image for sift extraction was not found. """
    pass

# TODO(sanchom): Test this function.
def build_extraction_list(caltech_image_directory, target_data_directory):
    """ Builds a list of image files and associated sift directories
    
    This list can be used by the sift extractor as images for extraction and
    output directories.

    Builds a list of pairs of (original_image, target_directory).
    If original_image == caltech_image_directory/obj_class/image_0001.jpg, then
    target_directory == target_data_directory/obj_class
    """
    extraction_list = []
    image_list = glob.glob(os.path.join(caltech_image_directory, '*/*.jpg'))
    image_list.extend(glob.glob(os.path.join(caltech_image_directory,
                                             '*/*.png')))
    for image_path in image_list:
        full_image_path = os.path.join(caltech_image_directory, image_path)
        class_name = \
            os.path.dirname(image_path).replace(caltech_image_directory,
                                                "").lstrip('/')
        target_data_path = os.path.join(target_data_directory, class_name)
        extraction_list.append((full_image_path, target_data_path))
    return extraction_list

# TODO (sanchom): test this function
def do_extraction_on_list(extraction_list, extraction_parameters=None,
                          num_processes=1):
    """ Performs sift extraction on the files in the extraction list.

    extraction_list is a list of tuples (image_path, destination_dir)
    that direct the sift extractor to extract sift from the image_path
    and store the result at destination_dir.

    extraction_parameters is a sift.ExtractionParameters protocol
    buffer.
    """
    if (extraction_parameters is None):
        raise RuntimeError('extraction_parameters needs to be set')

    # Check paths for extraction app and destination directories
    extract_descriptors_cli = 'extract_descriptors_cli'
    for (_, directory) in extraction_list:
        if (not os.path.exists(directory)):
            os.makedirs(directory)

    # This prepares a list of tuples
    # (original_image, extraction_path, extraction_parameters)
    unpacked = zip(*extraction_list)
    extraction_tuples = \
        zip(unpacked[0], unpacked[1],
            itertools.repeat(extraction_parameters.SerializeToString()))

    # Use multiple cores to do the extraction
    pool = multiprocessing.Pool(processes=num_processes)
    pool.map_async(do_extraction, extraction_tuples)
    pool.close()
    pool.join()

def split_into_train_test_random(file_list, num_train=None, num_test=None):
    """ Splits the file_list into a set of training and testing files.

    Returns a pair of lists (training_files, testing_files)

    Keywords arguments:
    num_train -- the number of files to use for training
    num_test -- the number of files to use for testing

    This just picks num_train randomly and num_test randomly such that
    there is no overlap between the two.
    """
    if num_train + num_test > len(file_list):
        raise TrainSizeError('num_train=%d + num_test=%d '
                             'is larger than len(file_list)=%d' %
                             (num_train, num_test, len(file_list)))
    selected = random.sample(file_list, num_train + num_test)
    return (selected[:num_train], selected[num_train:])

def do_extraction(extraction_tuple):
    """ Extracts sift from an image to a directory with given parameters.

    Runs the extract_descriptors_cli on an image with output to the
    specified directory and the parameters as given.

    Argument:
    - extraction_tuple: a 3-tuple (image, output_directory,
    extraction_parameters) with the
    absolute image path and absolute path of the output directory in
    which to place the extracted descriptor set
    (requested_parameters is a string-serialized protobuf)
    """
    (image, directory, requested_parameters_string) = extraction_tuple
    requested_parameters = sift_descriptors_pb2.ExtractionParameters()
    requested_parameters.ParseFromString(requested_parameters_string)
    if not os.path.exists(image):
        raise ImageNotFoundError('%s does not exist' % image)
    extract_descriptors_cli = 'extract_descriptors_cli'

    # First, we check if there's a file in the expected output location already
    # (ie. directory/image_basename.sift) with the requested parameters
    expected_output_path = \
        os.path.join(directory,
                     os.path.splitext(os.path.basename(image))[0] + '.sift')

    need_fresh_extraction = False
    try:
        existing_parameters = \
            sift_util.get_extraction_parameters(expected_output_path)
        # This complicated logic is needed because protocol buffers
        # don't implement an equality operator, and there are some
        # floating point errors introduced through in the message
        # chain that should be considered irrelevant.
        eps = 0.00001
        if ((existing_parameters.rotation_invariance !=
             requested_parameters.rotation_invariance) or
            (math.fabs(existing_parameters.normalization_threshold -
                       requested_parameters.normalization_threshold) > eps) or
            (existing_parameters.discard_unnormalized !=
             requested_parameters.discard_unnormalized) or
            (existing_parameters.multiscale !=
             requested_parameters.multiscale) or
            (math.fabs(existing_parameters.percentage -
                       requested_parameters.percentage) > eps) or
            (math.fabs(existing_parameters.minimum_radius -
                       requested_parameters.minimum_radius) > eps) or
            (existing_parameters.fractional_xy !=
             requested_parameters.fractional_xy) or
            (existing_parameters.top_left_x !=
             requested_parameters.top_left_x) or
            (existing_parameters.top_left_y !=
             requested_parameters.top_left_y) or
            (existing_parameters.bottom_right_x !=
             requested_parameters.bottom_right_x) or
            (existing_parameters.bottom_right_y !=
             requested_parameters.bottom_right_y) or
            (existing_parameters.implementation !=
             requested_parameters.implementation) or
            (existing_parameters.grid_method !=
             requested_parameters.grid_method) or
            (existing_parameters.smoothed !=
             requested_parameters.smoothed) or
            (math.fabs(existing_parameters.first_level_smoothing -
                       requested_parameters.first_level_smoothing) > eps) or
            (existing_parameters.fast !=
             requested_parameters.fast)
            ):
            need_fresh_extraction = True
    except IOError as e:
        need_fresh_extraction = True

    if (requested_parameters.grid_method ==
        sift_descriptors_pb2.ExtractionParameters.FIXED_3X3):
        grid_method_string = 'FIXED_3X3'
    elif (requested_parameters.grid_method ==
          sift_descriptors_pb2.ExtractionParameters.FIXED_8X8):
        grid_method_string = 'FIXED_8X8'
    elif (requested_parameters.grid_method ==
          sift_descriptors_pb2.ExtractionParameters.SCALED_3X3):
        grid_method_string = 'SCALED_3X3'
    elif (requested_parameters.grid_method ==
          sift_descriptors_pb2.ExtractionParameters.SCALED_BIN_WIDTH):
        grid_method_string = 'SCALED_BIN_WIDTH'
    elif (requested_parameters.grid_method ==
          sift_descriptors_pb2.ExtractionParameters.SCALED_DOUBLE_BIN_WIDTH):
        grid_method_string = 'SCALED_DOUBLE_BIN_WIDTH'

    command = (("%s --first_level_smoothing %f "
                "--percentage %f --clobber %s "
                "--normalization_threshold %f %s "
                "--minimum_radius %f %s %s %s "
                "--grid_type %s --output_directory %s %s "
                "--logtostderr") %
               (extract_descriptors_cli,
                requested_parameters.first_level_smoothing,
                requested_parameters.percentage,
                ('--discard' if
                 requested_parameters.discard_unnormalized else '--nodiscard'),
                requested_parameters.normalization_threshold,
                '--smooth' if requested_parameters.smoothed else '--nosmooth',
                requested_parameters.minimum_radius,
                ('--fractional_location' if
                 requested_parameters.fractional_xy else
                 '--nofractional_location'),
                ('--multiscale' if requested_parameters.multiscale else
                 '--nomultiscale'),
                '--fast' if requested_parameters.fast else '--nofast',
                grid_method_string,
                directory,
                image))

    if need_fresh_extraction:
        output = subprocess.Popen(command, shell=True,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE).communicate()
        print output[0],
        print output[1],
    else:
        print ('%s already exists with requested parameters' %
               expected_output_path)
