# Copyright (c) 2010, Sancho McCann, Wendi Zhuang

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

from optparse import OptionParser
import os
import sys

from naive_bayes_nearest_neighbor import caltech_util
from sift import sift_descriptors_pb2
from sift import sift_util

def main():
    parser = OptionParser()
    # This option points to the root directory of the image
    # dataset. Under the root directory should be category
    # directories, one per category, with images inside of them.
    parser.add_option('--dataset_path',
                      dest='dataset_path', default='',
                      help='A path to the root directory of the image dataset.')
    parser.add_option('--process_limit',
                      type='int', dest='process_limit', default=1,
                      help='The number of processors to use during extraction.')
    # These sift_ parameters are passed onto the sift extractor.
    parser.add_option('--sift_normalization_threshold',
                      type='float',
                      dest='sift_normalization_threshold',
                      default=2.0)
    parser.add_option('--sift_discard_unnormalized',
                      action='store_true',
                      dest='sift_discard_unnormalized',
                      default=True)
    parser.add_option('--sift_multiscale',
                      action='store_true',
                      dest='sift_multiscale',
                      default=False)
    parser.add_option('--sift_minimum_radius',
                      type='int',
                      dest='sift_minimum_radius',
                      default=0)
    parser.add_option('--sift_grid_type',
                      type='string',
                      dest='sift_grid_type',
                      default='FIXED_3X3')
    parser.add_option('--sift_first_level_smoothing',
                      type='float',
                      dest='sift_first_level_smoothing',
                      default=0.66)
    parser.add_option('--sift_nosmooth',
                      action='store_false',
                      dest='sift_smoothed',
                      default=True)
    parser.add_option('--sift_fast',
                      action='store_true',
                      dest='sift_fast',
                      default=True)
    parser.add_option('--sift_slow',
                      action='store_false',
                      dest='sift_fast')

    # TODO(WenDi): This should be a positional argument, as it is mandatory.
    parser.add_option('--features_directory',
                      action='store', type='string', dest='features_directory')

    (options, args) = parser.parse_args()
    print options

    # Move the sift_ command line arguments into an ExtactionParameters protobuf
    extraction_parameters = sift_descriptors_pb2.ExtractionParameters()
    extraction_parameters.normalization_threshold = \
        options.sift_normalization_threshold
    extraction_parameters.discard_unnormalized = \
        options.sift_discard_unnormalized
    extraction_parameters.multiscale = options.sift_multiscale
    extraction_parameters.minimum_radius = options.sift_minimum_radius
    extraction_parameters.first_level_smoothing = \
        options.sift_first_level_smoothing
    extraction_parameters.smoothed = \
        options.sift_smoothed
    extraction_parameters.fractional_xy = True
    extraction_parameters.fast = options.sift_fast
    if (options.sift_grid_type == 'FIXED_3X3'):
        extraction_parameters.grid_method = \
            sift_descriptors_pb2.ExtractionParameters.FIXED_3X3
    elif (options.sift_grid_type == 'SCALED_3X3'):
        extraction_parameters.grid_method = \
            sift_descriptors_pb2.ExtractionParameters.SCALED_3X3
    elif (options.sift_grid_type == 'SCALED_BIN_WIDTH'):
        extraction_parameters.grid_method = \
            sift_descriptors_pb2.ExtractionParameters.SCALED_BIN_WIDTH
    elif (options.sift_grid_type == 'SCALED_DOUBLE_BIN_WIDTH'):
        extraction_parameters.grid_method = \
            sift_descriptors_pb2.ExtractionParameters.SCALED_DOUBLE_BIN_WIDTH
    else:
        sys.stderr.write('Invalid --sift_grid_type.\n')
        sys.exit(1)

    # Get lists of extractions to do.
    caltech_data = options.dataset_path
    extraction_list = caltech_util.build_extraction_list(
        caltech_data, options.features_directory)
    caltech_util.do_extraction_on_list(extraction_list,
                                       extraction_parameters,
                                       options.process_limit)

if __name__ == '__main__':
    main()
