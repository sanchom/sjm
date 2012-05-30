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

#include "naive_bayes_nearest_neighbor/merged_classifier.h"

#include "glog/logging.h"
#include "gtest/gtest.h"

#include "flann/flann.hpp"

#include "sift/sift_descriptors.pb.h"
#include "sift/sift_util.h"

class MergedClassifierTest : public ::testing::Test {
};

TEST_F(MergedClassifierTest,
       TestConstructor) {
  sjm::nbnn::MergedClassifier classifier;
  int nearest_neighbors = 10;
  float alpha = 0;
  int checks = 32;
  int trees = 4;
  classifier.SetClassifierParams(nearest_neighbors, nearest_neighbors + 1,
                                 alpha, checks, trees);
  ASSERT_EQ(0, classifier.DataSize());
}

TEST_F(MergedClassifierTest,
       TestAddData) {
  sjm::nbnn::MergedClassifier classifier;
  int nearest_neighbors = 10;
  float alpha = 0;
  int checks = 32;
  int trees = 4;
  classifier.SetClassifierParams(nearest_neighbors, nearest_neighbors + 1,
                                 alpha, checks, trees);
  sjm::sift::DescriptorSet faces_descriptors;
  sjm::sift::ReadDescriptorSetFromFile(
      "../naive_bayes_nearest_neighbor/test_data/caltech_faces_set.sift",
      &faces_descriptors);
  sjm::sift::DescriptorSet emu_descriptors;
  sjm::sift::ReadDescriptorSetFromFile(
      "../naive_bayes_nearest_neighbor/test_data/caltech_emu_set.sift",
      &emu_descriptors);
  classifier.AddData("Faces", faces_descriptors);
  classifier.AddData("Emu", emu_descriptors);
  ASSERT_EQ(5596 + 5471, classifier.DataSize());
}

TEST_F(MergedClassifierTest,
       TestDieWhenClassifyBeforeBuild) {
  sjm::nbnn::MergedClassifier classifier;
  sjm::sift::DescriptorSet faces_descriptors;
  sjm::sift::ReadDescriptorSetFromFile(
      "../naive_bayes_nearest_neighbor/test_data/caltech_faces_set.sift",
      &faces_descriptors);
  sjm::sift::DescriptorSet emu_descriptors;
  sjm::sift::ReadDescriptorSetFromFile(
      "../naive_bayes_nearest_neighbor/test_data/caltech_emu_set.sift",
      &emu_descriptors);
  classifier.SetClassifierParams(5, 5 + 1, 1.5, 32, 2);
  classifier.AddData("Faces", faces_descriptors);
  classifier.AddData("Emu", emu_descriptors);
  sjm::sift::DescriptorSet query;
  for (int i = 0; i < 5; ++i) {
    sjm::sift::SiftDescriptor* d = query.add_sift_descriptor();
    d->CopyFrom(faces_descriptors.sift_descriptor(i));
  }
  ASSERT_DEATH(classifier.Classify(query, 1.0), ".*");
}

TEST_F(MergedClassifierTest,
       TestClassifyWorks) {
  sjm::nbnn::MergedClassifier classifier;
  classifier.SetClassifierParams(5, 5 + 1, 1.5, 32, 2);
  sjm::sift::DescriptorSet faces_descriptors;
  sjm::sift::ReadDescriptorSetFromFile(
      "../naive_bayes_nearest_neighbor/test_data/caltech_faces_set.sift",
      &faces_descriptors);
  sjm::sift::DescriptorSet emu_descriptors;
  sjm::sift::ReadDescriptorSetFromFile(
      "../naive_bayes_nearest_neighbor/test_data/caltech_emu_set.sift",
      &emu_descriptors);
  classifier.AddData("Faces", faces_descriptors);
  classifier.AddData("Emu", emu_descriptors);
  classifier.BuildIndex();
  sjm::sift::DescriptorSet faces_query;
  for (int i = 0; i < 5; ++i) {
    sjm::sift::SiftDescriptor* d = faces_query.add_sift_descriptor();
    d->CopyFrom(faces_descriptors.sift_descriptor(i));
  }
  sjm::nbnn::Result result = classifier.Classify(faces_query, 1.0);
  ASSERT_EQ(result.category, "Faces");

  sjm::sift::DescriptorSet emu_query;
  for (int i = 0; i < 5; ++i) {
    sjm::sift::SiftDescriptor* d = emu_query.add_sift_descriptor();
    d->CopyFrom(emu_descriptors.sift_descriptor(i));
  }
  sjm::nbnn::Result result2 = classifier.Classify(emu_query, 1.0);
  ASSERT_EQ(result2.category, "Emu");
}

// TODO(sanchom): Test checks for insertion of inconsistent
// descriptors.

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
