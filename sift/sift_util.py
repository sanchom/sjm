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

import os
import random
import struct
import time

import numpy

import sift_descriptors_pb2

def get_extraction_parameters(filename):
    parameters = sift_descriptors_pb2.ExtractionParameters()
    f = open(filename, "rb")
    (parameter_size,) = struct.unpack('i', f.read(4))
    (parameter_string,) = struct.unpack('%ds' % parameter_size,
                                        f.read(parameter_size))
    parameters.ParseFromString(parameter_string)
    f.close()
    return parameters

def load_descriptors(filename):
    descriptors = sift_descriptors_pb2.DescriptorSet()
    f = open(filename, "rb")
    (parameter_size,) = struct.unpack('i', f.read(4))
    f.seek(parameter_size, 1)  # Skip over the parameters
    (descriptor_size,) = struct.unpack('i', f.read(4))
    (descriptor_string,) = struct.unpack('%ds' % descriptor_size,
                                         f.read(descriptor_size))
    descriptors.ParseFromString(descriptor_string)
    return descriptors

def convert_bare_set_to_set_with_params(original_file, destination_file):
    descriptor_set = sift_descriptors_pb2.DescriptorSet()
    descriptor_set.ParseFromString(original_file.read())
    serialized_parameters = descriptor_set.parameters.SerializeToString()
    serialized_descriptors = descriptor_set.SerializeToString()
    parameters_size = len(serialized_parameters)
    descriptors_size = len(serialized_descriptors)

    data = struct.pack('i', parameters_size)
    destination_file.write(data)
    destination_file.write(serialized_parameters)
    data = struct.pack('i', descriptors_size)
    destination_file.write(data)
    destination_file.write(serialized_descriptors)

def count_descriptors_in_file(filename):
    descriptor_set = load_descriptors(filename)
    return len(descriptor_set.sift_descriptor)

def count_descriptors_in_list(file_list):
    count = 0
    for f in file_list:
        count += count_descriptors_in_file(f)
    return count

def merge_descriptor_sets(descriptor_set_list):
    """ Returns a merged descriptor set from the given list.
    """
    merged_descriptor_set = sift_descriptors_pb2.DescriptorSet()
    for descriptor_set in descriptor_set_list:
        for descriptor in descriptor_set.sift_descriptor:
            new_descriptor = merged_descriptor_set.sift_descriptor.add()
            new_descriptor.x = descriptor.x
            new_descriptor.y = descriptor.y
            new_descriptor.scale = descriptor.scale
            new_descriptor.bin.extend(descriptor.bin)

        merged_descriptor_set.parameters.CopyFrom(descriptor_set.parameters)

    return merged_descriptor_set

def convert_protobuf_descriptor_to_weighted_array(descriptor, alpha=0):
    """ Converts an instance of SiftDescriptor to a numpy array

    This throws out the location information unless it's added using the
    alpha weighting as extra dimensions at the end.

    Arguments:
    -- descriptor: an instance of sift_descriptors_pb2.SiftDescriptor
    -- alpha: an optional weighting to add the spatial dimensions onto the
       end of the numpy array

    Returns: a 1d umpy array of shape (dimensions,) or (dimensions+2,) if
    alpha > 0
    """
    dimensions = len(descriptor.bin) + (0 if alpha == 0 else 2)
    descriptor_array = numpy.zeros(dimensions, numpy.uint8)
    descriptor_array[:len(descriptor.bin)] = numpy.array(descriptor.bin)
    # Optionally, include the descriptor location as extra dimensions
    # The indices into which we're doing the lookup should have been
    # built with this same convention.
    if alpha > 0:
        descriptor_array[-2] = descriptor.x * 127 * alpha + 0.5
        descriptor_array[-1] = descriptor.y * 127 * alpha + 0.5

    return descriptor_array

def load_array_from_files(file_list, alpha=0, max_points=None, verbose=False):
    """ Loads sift descriptors from a list of files and returns an array.

    Returns a numpy array of shape (num_descriptors, num_dimensions) of
    all the sift descriptors (up to max_points) in the file list.

    If max_points < the number of points available, a random sampling
    of the available points is taken to return approximately max_points.

    If max_points > the number of points available, all points are
    included in the array.

    If alpha > 0, includes weighted location information at the end of the
    descriptor in two extra dimensions.

    This runs much slower when max_points is specified. Points are taken
    uniformly at random from the entire file_list, so a count of available
    descriptors needs to be done before again proceeding through the list.

    Verbose prints progress
    """
    initial_size = 500
    # Set the array size to its maximum or the initial size
    array_size = max_points if max_points is not None else initial_size
    dimensions = 128 if alpha == 0 else 130
    points = numpy.zeros((array_size, dimensions),
                         numpy.uint8)
    p_index = 0

    # TODO (sanchom): Don't count and then parse, simple parse once and count
    percentage_to_load = 1
    if max_points is not None:
        num_descriptors = count_descriptors_in_list(file_list)
        if num_descriptors > 0:
            percentage_to_load = min(1, float(max_points) / num_descriptors)
        else:
            percentage_to_load = 0

    start_time = time.time()
    for count, sift_filename in enumerate(file_list):
        sift_filename = os.path.expanduser(sift_filename)
        descriptor_set = load_descriptors(sift_filename)
        for d in descriptor_set.sift_descriptor:
            if max_points is None or p_index < max_points:
                if random.random() < percentage_to_load:
                    # Double size of array if running past the end
                    if p_index >= len(points):
                        points.resize((len(points) * 2, dimensions))
                    points[p_index,:128] = numpy.array(d.bin, numpy.uint8)
                    if alpha > 0:
                        points[p_index,-2] = d.x * 127 * alpha + 0.5
                        points[p_index,-1] = d.y * 127 * alpha + 0.5
                    p_index += 1
        now = time.time()
        spf = (now - start_time) / (count + 1)
        if verbose:
            print 'File %d of %d, averaging %f seconds per file' % (count + 1, len(file_list), spf)
    points = points[0:p_index,:]
    return points
