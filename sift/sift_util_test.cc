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

// This file tests c++ descriptor utilities, such as those for writing
// protocol buffers to file and reading them back.

// File under test.
#include "sift_util.h"

// STL includes
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>

// Third party includes.
#include "gtest/gtest.h"

// My includes.
#include "sift_descriptors.pb.h"

using namespace std;

class SiftUtilTest : public ::testing::Test {
 protected:
  void SetUp() {
    filename_ = "/tmp/test.sift";
    remove(filename_.c_str());
    sjm::sift::ExtractionParameters *parameters =
        descriptors_.mutable_parameters();
    parameters->set_fractional_xy(true);
    parameters->set_multiscale(true);
    parameters->set_minimum_radius(16);
    parameters->set_implementation(sjm::sift::ExtractionParameters::VLFEAT);

    // Next, we add two descriptors into the message
    sjm::sift::SiftDescriptor *d = descriptors_.add_sift_descriptor();
    for (int i = 0; i < 128; ++i) {
      d->add_bin(i);
    }
    d = descriptors_.add_sift_descriptor();
    for (int i = 0; i < 128; ++i) {
      d->add_bin(127-i);
    }
  }

  sjm::sift::DescriptorSet descriptors_;
  string filename_;
};

TEST_F(SiftUtilTest, CanWriteFile) {
  string filename = "/tmp/test.sift";
  sjm::sift::WriteDescriptorSetToFile(descriptors_, filename_);
  // Checks that the file exists by seeing if we can open it.
  ifstream test_file(filename_.c_str());
  ASSERT_TRUE(test_file.is_open());
}

TEST_F(SiftUtilTest, SavedFileContainsParameters) {
  string filename = "/tmp/test.sift";
  sjm::sift::WriteDescriptorSetToFile(descriptors_, filename_);
  ifstream test_file(filename_.c_str());
  int parameter_size;
  test_file.read((char*)&parameter_size,
                 sizeof(parameter_size));
  ASSERT_EQ(11, parameter_size);
  char *parameter_buffer = new char[parameter_size];
  test_file.read(parameter_buffer, parameter_size);
  string parameter_string(parameter_buffer, parameter_size);
  delete[] parameter_buffer;
  ASSERT_TRUE(test_file.good());
  sjm::sift::ExtractionParameters parameters;
  parameters.ParseFromString(parameter_string);
  ASSERT_TRUE(parameters.fractional_xy());
  ASSERT_TRUE(parameters.multiscale());
  ASSERT_EQ(16, parameters.minimum_radius());
  ASSERT_EQ(sjm::sift::ExtractionParameters::VLFEAT,
            parameters.implementation());
}

TEST_F(SiftUtilTest, SavedFileContainsDescriptors) {
  string filename = "/tmp/test.sift";
  sjm::sift::WriteDescriptorSetToFile(descriptors_, filename_);
  ifstream test_file(filename_.c_str());
  int parameter_size;
  test_file.read((char*)&parameter_size, sizeof(parameter_size));
  test_file.seekg(parameter_size, ios_base::cur);
  int descriptors_size;
  test_file.read((char*)&descriptors_size,
                 sizeof(descriptors_size));
  char *descriptors_buffer = new char[descriptors_size];
  test_file.read(descriptors_buffer, descriptors_size);
  string descriptors_string(descriptors_buffer, descriptors_size);
  delete[] descriptors_buffer;
  ASSERT_TRUE(test_file.good());
  sjm::sift::DescriptorSet descriptors;
  ASSERT_TRUE(descriptors.ParseFromString(descriptors_string));
  ASSERT_EQ(2, descriptors.sift_descriptor_size());
  ASSERT_EQ(0, descriptors.sift_descriptor(0).bin(0));
  ASSERT_EQ(1, descriptors.sift_descriptor(0).bin(1));
  ASSERT_EQ(127, descriptors.sift_descriptor(0).bin(127));
  ASSERT_EQ(127, descriptors.sift_descriptor(1).bin(0));
  ASSERT_EQ(126, descriptors.sift_descriptor(1).bin(1));
  ASSERT_EQ(0, descriptors.sift_descriptor(1).bin(127));
}

TEST_F(SiftUtilTest, ParametersReaderWorks) {
  string filename = "/tmp/test.sift";
  sjm::sift::WriteDescriptorSetToFile(descriptors_, filename_);
  sjm::sift::ExtractionParameters parameters;
  sjm::sift::ReadParametersFromFile(filename_, &parameters);
  ASSERT_TRUE(parameters.fractional_xy());
  ASSERT_TRUE(parameters.multiscale());
  ASSERT_EQ(16, parameters.minimum_radius());
  ASSERT_EQ(sjm::sift::ExtractionParameters::VLFEAT,
            parameters.implementation());  
}

TEST_F(SiftUtilTest, DescriptorSetReaderWorks) {
  string filename = "/tmp/test.sift";
  sjm::sift::WriteDescriptorSetToFile(descriptors_, filename_);
  sjm::sift::DescriptorSet descriptors;
  sjm::sift::ReadDescriptorSetFromFile(filename_, &descriptors);
  ASSERT_TRUE(descriptors.parameters().fractional_xy());
  ASSERT_TRUE(descriptors.parameters().multiscale());
  ASSERT_EQ(16, descriptors.parameters().minimum_radius());
  ASSERT_EQ(sjm::sift::ExtractionParameters::VLFEAT,
            descriptors.parameters().implementation());  
  ASSERT_EQ(2, descriptors.sift_descriptor_size());
  ASSERT_EQ(0, descriptors.sift_descriptor(0).bin(0));
  ASSERT_EQ(1, descriptors.sift_descriptor(0).bin(1));
  ASSERT_EQ(127, descriptors.sift_descriptor(0).bin(127));
  ASSERT_EQ(127, descriptors.sift_descriptor(1).bin(0));
  ASSERT_EQ(126, descriptors.sift_descriptor(1).bin(1));
  ASSERT_EQ(0, descriptors.sift_descriptor(1).bin(127));

}

TEST_F(SiftUtilTest,
       TestProtobufToArrayConversion) {
  sjm::sift::SiftDescriptor descriptor;
  descriptor.add_bin(15);
  descriptor.add_bin(35);
  descriptor.set_x(0.2);
  descriptor.set_y(0.3);
  descriptor.set_scale(1);
  uint8_t descriptor_array[4];
  int length =
      sjm::sift::ConvertProtobufDescriptorToWeightedArray(descriptor, 0,
                                                          descriptor_array);
  ASSERT_EQ(2, length);
  ASSERT_FLOAT_EQ(15, descriptor_array[0]);
  ASSERT_FLOAT_EQ(35, descriptor_array[1]);

  float alpha = 0.5;
  length =
      sjm::sift::ConvertProtobufDescriptorToWeightedArray(descriptor, alpha,
                                                          descriptor_array);
  ASSERT_EQ(4, length);
  ASSERT_FLOAT_EQ(15, descriptor_array[0]);
  ASSERT_FLOAT_EQ(35, descriptor_array[1]);
  ASSERT_FLOAT_EQ(static_cast<int>(0.2 * 127 * alpha + 0.5),
                  descriptor_array[2]);
  ASSERT_FLOAT_EQ(static_cast<int>(0.3 * 127 * alpha + 0.5),
                  descriptor_array[3]);

  alpha = 1.5;
  length =
      sjm::sift::ConvertProtobufDescriptorToWeightedArray(descriptor, alpha,
                                                          descriptor_array);
  ASSERT_EQ(4, length);
  ASSERT_FLOAT_EQ(15, descriptor_array[0]);
  ASSERT_FLOAT_EQ(35, descriptor_array[1]);
  ASSERT_FLOAT_EQ(static_cast<int>(0.2 * 127 * alpha + 0.5),
                  descriptor_array[2]);
  ASSERT_FLOAT_EQ(static_cast<int>(0.3 * 127 * alpha + 0.5),
                  descriptor_array[3]);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
