# Copyright 2011 Sancho McCann

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

# This script runs the Spatial Pyramid and Spatially Local Coding
# experiments published in "Spatially Local Coding for Object
# Recognition".

import glob
import logging
from optparse import OptionParser
import os
import subprocess
import sys
import time

from naive_bayes_nearest_neighbor import caltech_util
from sift import sift_descriptors_pb2

def main():
    parser = OptionParser()
    # This option points to the root directory of the image
    # dataset. Under the root directory should be category
    # directories, one per category, with images inside of them.
    parser.add_option('--dataset_path',
                      dest='dataset_path', default='',
                      help='A path to the root directory of the image dataset.')
    parser.add_option('-n', '--num_categories',
                      type='int', dest='num_categories', default=None,
                      help=('The number of categories to test on. '
                            'If unspecified, use all of them.'))
    parser.add_option('--work_directory',
                      type='string', dest='work_directory', default='',
                      help='The directory to put all intermediate files.')
    parser.add_option('-o', '--results-file',
                      type='string', dest='output', default='results.txt',
                      help='The file to store the accuracy results in')
    parser.add_option('--kernel',
                      type='string', dest='kernel', default='',
                      help='The kernel to use for the spatial pyramids.')
    parser.add_option('--caltech_256',
                      action='store_true', dest='caltech_256', default=False)
    parser.add_option('--scenes',
                      action='store_true', dest='scenes', default=False)
    parser.add_option('--c',
                      type='float', dest='c', default=0)
    parser.add_option('--clobber',
                      action='store_true',
                      dest='clobber', default=False)
    parser.add_option('--noclobber',
                      action='store_false',
                      dest='clobber', default=False)
    # These sift_ parameters are passed onto the sift extractor.
    parser.add_option('--sift_normalization_threshold',
                      type='float',
                      dest='sift_normalization_threshold',
                      default=0.5)
    parser.add_option('--sift_discard_unnormalized',
                      action='store_true',
                      dest='sift_discard_unnormalized',
                      default=False)
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
                      default=0.5)
    parser.add_option('--features_directory',
                      action='store', type='string', dest='features_directory')
    parser.add_option('--process_limit',
                      action='store', type='int', default=1,
                      dest='process_limit')
    parser.add_option('--num_train',
                      action='store', type='int', dest='num_train')
    parser.add_option('--num_test',
                      action='store', type='int', dest='num_test')
    parser.add_option('--codeword_locality',
                      action='store', type='int', dest='codeword_locality')
    parser.add_option('--pooling',
                      action='store', type='string', dest='pooling')
    parser.add_option('--dictionary_training_size',
                      action='store', type='int', default=1000000,
                      dest='dictionary_training_size')
    parser.add_option('--clobber_dictionary',
                      action='store_true', default=False,
                      dest='clobber_dictionary')
    parser.add_option('--pyramid_levels',
                      type='int', default=3,
                      dest='pyramid_levels')
    parser.add_option('--single_level',
                      type='int', default=-1,
                      dest='single_level')
    parser.add_option('--dictionary',
                      action="append",
                      type='string', default=[],
                      dest='dictionary')
    parser.add_option('--kmeans_accuracy', action='store',
                      type='float', dest='kmeans_accuracy',
                      default=1)

    (options, args) = parser.parse_args()
    print options

    if (len(options.dictionary) == 0):
        sys.stderr.write('Dictionary argument not provided.\n')
        sys.exit(1)

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
    extraction_parameters.fractional_xy = True
    extraction_parameters.fast = True
    if (options.sift_grid_type == 'FIXED_3X3'):
        extraction_parameters.grid_method = \
            sift_descriptors_pb2.ExtractionParameters.FIXED_3X3
    elif (options.sift_grid_type == 'FIXED_8X8'):
        extraction_parameters.grid_method = \
            sift_descriptors_pb2.ExtractionParameters.FIXED_8X8
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

    extraction_list = caltech_util.build_extraction_list(
        options.dataset_path, options.features_directory)
    caltech_util.do_extraction_on_list(extraction_list,
                                       extraction_parameters,
                                       options.process_limit)
    if (not os.path.exists(options.work_directory)):
        os.mkdir(options.work_directory)

    training_lists = {}
    testing_lists = {}

    object_categories = list(set(os.listdir(options.features_directory)) -
                             set(['BACKGROUND_Google']) -
                             set(['257.clutter']))

    if options.num_categories is None:
        options.num_categories = len(object_categories)

    dictionary_params = []

    for d in options.dictionary:
        (dimensions, alpha) = d.split(':')
        logging.info('dictionary: %d dimensions, %0.2f alpha' %
                     (int(dimensions), float(alpha)))
        dictionary_path = os.path.join(options.work_directory,
                                       'caltech_%dd_%0.2fa.dictionary' %
                                       (int(dimensions), float(alpha)))
        dictionary_params.append((dictionary_path,
                                  int(dimensions),
                                  float(alpha)))

    logging.info('Splitting Caltech data into training and testing.')
    for category in object_categories[:options.num_categories]:
        file_list = glob.glob(
            os.path.join(os.path.join(options.features_directory, category),
                         "*.sift"))
        try:
            capped_num_test = len(file_list) - options.num_train
            (training_list, testing_list) = \
                caltech_util.split_into_train_test_random(file_list,
                                                          options.num_train,
                                                          capped_num_test)
        except caltech_util.CaltechUtilError as err:
            print err
            sys.exit(0)
        training_lists[category] = training_list
        testing_lists[category] = testing_list

    descriptor_training_list = os.path.join(options.work_directory,
                                            'descriptor_training_list.txt')
    # Make the codeword_training_file if it doesn't exist.
    if (options.clobber or not os.path.exists(descriptor_training_list)):
        logging.info('Writing list of .sift training files.')
        codeword_training_file = open(descriptor_training_list, 'w')
        for category, training_list in training_lists.iteritems():
            for t in training_list:
                codeword_training_file.write('%s\n' % t)
        codeword_training_file.close()
    else:
        logging.info('Training file already exists, skipping.')

    training_file_list_reread = []
    for line in open(descriptor_training_list, 'r').readlines():
        training_file_list_reread.append(line.strip())

    # Make the descriptor testing file, if it doesn't exist.
    descriptor_testing_list = os.path.join(options.work_directory,
                                           'descriptor_testing_list.txt')
    if (options.clobber or not os.path.exists(descriptor_testing_list)):
        logging.info('Writing list of .sift testing files.')
        descriptor_testing_file = open(descriptor_testing_list, 'w')
        for category, testing_list in testing_lists.iteritems():
            num_test_found = 0
            for t in testing_list:
                if t not in training_file_list_reread:
                    descriptor_testing_file.write('%s\n' % t)
                    num_test_found += 1
                if num_test_found == options.num_test:
                    break
        descriptor_testing_file.close()
    else:
        logging.info('Testing file already exists, skipping.')

    # Create the first checkpoint.
    open(os.path.join(options.work_directory, 'checkpoint_a.txt'), 'w').close()

    # List for holding subprocess objects
    codebook_builders = []
    for dict_id, (dictionary_path, dimensions, alpha) in enumerate(dictionary_params):
        if (options.clobber_dictionary or not os.path.exists(dictionary_path)):
            logging.info('Creating dictionary.')
            codebook_cli_path = 'codebook_cli'
            command = ('%s --input list:%s --output %s --clusters %d '
                       '--max_descriptors %d --location_weighting %f '
                       '--initialization SUBSAMPLED_KMEANSPP --accuracy %f '
                       '--initialization_checkpoint_file %s '
                       '--logtostderr' %
                       (codebook_cli_path,
                        descriptor_training_list,
                        dictionary_path,
                        dimensions,
                        options.dictionary_training_size,
                        alpha,
                        options.kmeans_accuracy,
                        os.path.join(options.work_directory,
                                     'checkpoint_b_%d.txt' % dict_id)
                        ))
            # TODO(sanchom): Limit the number of simultaneous
            # subprocesses to options.process_limit.
            #
            # Create a popen object for this subprocess
            p = subprocess.Popen(command, shell=True)
            codebook_builders.append(p)
        else:
            logging.info('Dictionary already exists at %s, skipping. '
                         'Use --clobber_dictionary to force a rebuild.' % dictionary_path)
    # Wait for all codebook subprocesses to finish, and checkpoint when they're done
    finished_processes = []
    while len(finished_processes) != len(codebook_builders):
        for dict_id, p in enumerate(codebook_builders):
            # If the process is newly finished,
            if (not dict_id in finished_processes) and (p.poll() is not None):
                # Checkpoint it.
                open(os.path.join(options.work_directory,
                                  'checkpoint_c_%d.txt' % dict_id), 'w').close()
                # Mark this completion as known.
                finished_processes.append(dict_id)
                break
        time.sleep(0.1)

    # Convert the training files to pyramid representations.
    logging.info('Converting training files from %s to pyramid representations.' % descriptor_training_list)
    pyramid_cli_path = 'spatial_pyramid_cli'
    command = (
        '%s --codebooks %s --input list:%s --levels %d --single_level %d --k %d --pooling %s --thread_limit %d --logtostderr' %
        (pyramid_cli_path,
         ",".join([a for a, b, c in dictionary_params]),
         descriptor_training_list,
         options.pyramid_levels,
         options.single_level,
         options.codeword_locality,
         options.pooling,
         options.process_limit))
    p = subprocess.Popen(command, shell=True)
    p.wait()

    # Checkpoint end of training file conversion.
    open(os.path.join(options.work_directory, 'checkpoint_d.txt'), 'w').close()

    # Convert the testing files to pyramid representations.
    logging.info('Converting testing files from %s to pyramid representations.' % descriptor_testing_list)
    command = (
        '%s --codebooks %s --input list:%s --levels %d --single_level %d --k %d --pooling %s --thread_limit %d --logtostderr' %
        (pyramid_cli_path,
         ",".join([a for a, b, c in dictionary_params]),
         descriptor_testing_list,
         options.pyramid_levels,
         options.single_level,
         options.codeword_locality,
         options.pooling,
         options.process_limit))
    p = subprocess.Popen(command, shell=True)
    p.wait()

    # Checkpoint end of testing file conversion.
    open(os.path.join(options.work_directory, 'checkpoint_e.txt'), 'w').close()

    # Creating the pyramid lists for training and testing the svm models.
    pyramid_training_list = os.path.join(options.work_directory,
                                         'pyramid_training_list.txt')
    f = open(pyramid_training_list, 'w')
    for descriptor_file in open(descriptor_training_list).readlines():
        descriptor_file = descriptor_file.strip()
        pyramid_file = descriptor_file.replace('.sift', '.pyramid')
        # Determine the category
        category = 'None'
        for c in object_categories[:options.num_categories]:
            if os.path.splitext(pyramid_file)[0].find(c) != -1:
                category = c
        f.write('%s:%s\n' % (pyramid_file, category))
    f.close()

    pyramid_testing_list = os.path.join(options.work_directory,
                                        'pyramid_testing_list.txt')
    f = open(pyramid_testing_list, 'w')
    for descriptor_file in open(descriptor_testing_list).readlines():
        descriptor_file = descriptor_file.strip()
        pyramid_file = descriptor_file.replace('.sift', '.pyramid')
        category = 'None'
        for c in object_categories[:options.num_categories]:
            if os.path.splitext(pyramid_file)[0].find(c) != -1:
                category = c
        f.write('%s:%s\n' % (pyramid_file, category))
    f.close()

    # Train a model for each category.
    trainer_cli = 'trainer_cli'
    command = (
        '%s --training_list %s --kernel %s --output_directory /tmp '
        '--thread_limit %s --c %f --gram_matrix_checkpoint_file %s '
        '--cross_validation_checkpoint_file %s '
        '--logtostderr' %
        (trainer_cli,
         pyramid_training_list,
         options.kernel,
         options.process_limit,
         options.c,
         os.path.join(options.work_directory, 'checkpoint_f.txt'),
         os.path.join(options.work_directory, 'checkpoint_g.txt')
         ))
    p = subprocess.Popen(command, shell=True)
    p.wait()

    # Checkpoint end of training.
    open(os.path.join(options.work_directory, 'checkpoint_h.txt'), 'w').close()

    # Make the model list.
    model_list_path = os.path.join(options.work_directory, 'model_list.txt')
    model_list_file = open(model_list_path, 'w')
    model_list = glob.glob('/tmp/*.svm')
    for m in model_list:
        category = os.path.splitext(os.path.basename(m))[0]
        model_list_file.write('%s:%s\n' % (m, category))
    model_list_file.close()

    # Validate the models on the test data.
    validate_cli = 'validate_cli'
    command = \
        '%s --training_list %s --model_list %s --testing_list %s --kernel %s --thread_limit %d --result_file %s --logtostderr' % \
        (validate_cli,
         pyramid_training_list,
         model_list_path,
         pyramid_testing_list,
         options.kernel,
         options.process_limit,
         os.path.join(options.work_directory, options.output))
    p = subprocess.Popen(command, shell=True)
    p.wait()

    # Checkpoint end of validation.
    open(os.path.join(options.work_directory, 'checkpoint_i.txt'), 'w').close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
