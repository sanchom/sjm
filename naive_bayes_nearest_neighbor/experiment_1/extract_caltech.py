# Copyright 2010 Sancho McCann
# Author: Sancho McCann, Wendi Zhuang

from optparse import OptionParser
import os
import sys

from naive_bayes_nearest_neighbor import caltech_util
from naive_bayes_nearest_neighbor import nbnn_classifier
from sift import sift_descriptors_pb2
from sift import sift_util

def main():
    parser = OptionParser()
    parser.add_option('--caltech_256',
                      action='store_true', dest='caltech_256', default=False)
    parser.add_option('--caltech_101',
                      action='store_false', dest='caltech_256', default=False)
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
    if options.caltech_256:
        caltech_data = '/lci/project/lowe/sanchom/256_ObjectCategories-Boiman'
    else:
        caltech_data = '/lci/project/lowe/sanchom/101_ObjectCategories-Boiman'

    extraction_list = caltech_util.build_extraction_list(
        caltech_data, options.features_directory)
    caltech_util.do_extraction_on_list(extraction_list,
                                       extraction_parameters,
                                       options.process_limit)

if __name__ == '__main__':
    main()
