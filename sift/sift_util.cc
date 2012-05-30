// Copyright (c) 2011, Sancho McCann

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:

// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <iostream>
#include <fstream>
#include <string>

#include "boost/filesystem.hpp"

#include "glog/logging.h"

#include "sift/sift_descriptors.pb.h"
#include "util/util.h"

using std::ifstream;
using std::ios;
using std::ios_base;
using std::ofstream;
using std::string;

namespace sjm {
namespace sift {

void WriteDescriptorSetToFile(const sjm::sift::DescriptorSet &descriptors,
                              const string &filename) {
  ofstream output_file(filename.c_str(), ios::binary | ios::trunc);
  sjm::sift::ExtractionParameters parameters = descriptors.parameters();
  string serialized_parameters;
  string serialized_descriptors;
  parameters.SerializeToString(&serialized_parameters);
  descriptors.SerializeToString(&serialized_descriptors);
  int parameters_size = serialized_parameters.size();
  int descriptors_size = serialized_descriptors.size();
  output_file.write((char*)&parameters_size, sizeof(parameters_size));
  output_file << serialized_parameters;
  output_file.write((char*)&descriptors_size, sizeof(descriptors_size));
  output_file << serialized_descriptors;
  output_file.close();
}

void ReadParametersFromFile(const std::string &filename,
                            sjm::sift::ExtractionParameters *parameters) {
  ifstream input_file(filename.c_str());
  int parameter_size;
  input_file.read((char*)&parameter_size, sizeof(parameter_size));
  char *parameter_buffer = new char[parameter_size];
  input_file.read(parameter_buffer, parameter_size);
  string parameter_string(parameter_buffer, parameter_size);
  delete[] parameter_buffer;
  parameters->Clear();
  parameters->ParseFromString(parameter_string);
  input_file.close();
}

void ReadDescriptorSetFromFile(const std::string &filename,
                               sjm::sift::DescriptorSet *descriptors) {
  string expanded_filename = sjm::util::expand_user(filename);
  CHECK(boost::filesystem::exists(expanded_filename.c_str())) <<
      expanded_filename << " doesn't exist.";
  ifstream input_file(expanded_filename.c_str());
  int parameter_size;
  input_file.read((char*)&parameter_size, sizeof(parameter_size));
  input_file.seekg(parameter_size, ios_base::cur);
  int descriptor_size;
  input_file.read((char*)&descriptor_size, sizeof(descriptor_size));
  char *descriptor_buffer = new char[descriptor_size];
  input_file.read(descriptor_buffer, descriptor_size);
  string descriptor_string(descriptor_buffer, descriptor_size);
  delete[] descriptor_buffer;
  descriptors->Clear();
  descriptors->ParseFromString(descriptor_string);
  input_file.close();
}

int ConvertProtobufDescriptorToWeightedArray(
    const sjm::sift::SiftDescriptor &descriptor,
    const float alpha,
    uint8_t *destination) {
  int dimensions = descriptor.bin_size();
  if (alpha > 0) {
    dimensions += 2;
  }
  for (int i = 0; i < descriptor.bin_size(); ++i) {
    destination[i] = descriptor.bin(i);
  }
  if (alpha > 0) {
    destination[dimensions - 2] =
        static_cast<int>(descriptor.x() * 127 * alpha + 0.5);
    destination[dimensions - 1] =
        static_cast<int>(descriptor.y() * 127 * alpha + 0.5);
  }
  return dimensions;
}
}}  // namespaces
