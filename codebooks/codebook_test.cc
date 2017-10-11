// Copyright 2011 Sancho McCann

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

// This file tests the construction of codebooks from sift files.

// Files under test.
#include "codebooks/codebook_builder.h"

// STL includes.
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>
#include <utility>

// Third party includes.
#include "glog/logging.h"
#include "gtest/gtest.h"

#include "codebooks/dictionary.pb.h"
#include "sift/sift_descriptors.pb.h"
#include "sift/sift_util.h"
#include "util/util.h"

using std::make_pair;
using std::pair;
using std::string;
using std::vector;

class CodebookTestRealDataNoLocation : public ::testing::Test {
 protected:
  void SetUp() {
    string test_file_1(
        "../naive_bayes_nearest_neighbor/test_data/"
        "caltech_emu_set.sift");
    string test_file_2(
        "../naive_bayes_nearest_neighbor/test_data/"
        "caltech_faces_set.sift");
    sjm::sift::DescriptorSet descriptors;
    sjm::sift::ReadDescriptorSetFromFile(
        sjm::util::expand_user(test_file_1), &descriptors);
    builder_.AddData(descriptors, 1.0, 0.0f);
    sjm::sift::ReadDescriptorSetFromFile(
        sjm::util::expand_user(test_file_2), &descriptors);
    builder_.AddData(descriptors, 1.0, 0.0f);

    data_dimensions_ = descriptors.sift_descriptor(0).bin_size();
  }

  sjm::codebooks::CodebookBuilder builder_;
  int data_dimensions_;
};

class CodebookTestRealDataWithLocation : public ::testing::Test {
 protected:
  void SetUp() {
    string test_file_1(
        "../naive_bayes_nearest_neighbor/test_data/"
        "caltech_emu_set.sift");
    string test_file_2(
        "../naive_bayes_nearest_neighbor/test_data/"
        "caltech_faces_set.sift");
    const float location_weighting = 1.5;
    sjm::sift::DescriptorSet descriptors;
    sjm::sift::ReadDescriptorSetFromFile(
        sjm::util::expand_user(test_file_1), &descriptors);
    builder_.AddData(descriptors, 1.0, location_weighting);
    sjm::sift::ReadDescriptorSetFromFile(
        sjm::util::expand_user(test_file_2), &descriptors);
    builder_.AddData(descriptors, 1.0, location_weighting);

    data_dimensions_ = descriptors.sift_descriptor(0).bin_size() + 2;
  }

  sjm::codebooks::CodebookBuilder builder_;
  int data_dimensions_;
};

TEST_F(CodebookTestRealDataNoLocation,
       CodebookBuilderReturnsRequestedNumCentroids) {
  sjm::codebooks::Dictionary dictionary;

  builder_.Cluster(30, 11);
  builder_.GetDictionary(&dictionary);
  ASSERT_EQ(30, dictionary.centroid_size());

  builder_.Cluster(7, 11);
  builder_.GetDictionary(&dictionary);
  ASSERT_EQ(7, dictionary.centroid_size());
}

TEST_F(CodebookTestRealDataWithLocation,
       CodebookBuilderReturnsRequestedNumCentroids) {
  sjm::codebooks::Dictionary dictionary;

  builder_.Cluster(30, 11);
  builder_.GetDictionary(&dictionary);
  ASSERT_EQ(30, dictionary.centroid_size());

  builder_.Cluster(7, 11);
  builder_.GetDictionary(&dictionary);
  ASSERT_EQ(7, dictionary.centroid_size());
}

TEST_F(CodebookTestRealDataNoLocation,
       CodebookBuilderReturnsCentroidsWithLength) {
  sjm::codebooks::Dictionary dictionary;
  builder_.Cluster(7, 11);
  builder_.GetDictionary(&dictionary);
  for (int i = 0; i < dictionary.centroid_size(); ++i) {
    ASSERT_GT(dictionary.centroid(i).bin_size(), 0) <<
        "Found zero-length centroids.";
  }
}

TEST_F(CodebookTestRealDataWithLocation,
       CodebookBuilderReturnsCentroidsWithLength) {
  sjm::codebooks::Dictionary dictionary;
  builder_.Cluster(7, 11);
  builder_.GetDictionary(&dictionary);
  for (int i = 0; i < dictionary.centroid_size(); ++i) {
    ASSERT_GT(dictionary.centroid(i).bin_size(), 0) <<
        "Found zero-length centroids.";
  }
}

TEST_F(CodebookTestRealDataNoLocation,
       CodebookBuilderReturnsNonZeroCentroids) {
  sjm::codebooks::Dictionary dictionary;
  builder_.Cluster(7, 11);
  builder_.GetDictionary(&dictionary);
  for (int i = 0; i < dictionary.centroid_size(); ++i) {
    bool is_zero = true;
    for (int j = 0; j < dictionary.centroid(i).bin_size(); ++j) {
      if (dictionary.centroid(i).bin(j) != 0) {
        is_zero = false;
        break;
      }
    }
    ASSERT_FALSE(is_zero) << "Found zero descriptor.";
  }
}


TEST_F(CodebookTestRealDataWithLocation,
       CodebookBuilderReturnsNonZeroCentroids) {
  sjm::codebooks::Dictionary dictionary;
  builder_.Cluster(7, 11);
  builder_.GetDictionary(&dictionary);
  for (int i = 0; i < dictionary.centroid_size(); ++i) {
    bool is_zero = true;
    for (int j = 0; j < dictionary.centroid(i).bin_size(); ++j) {
      if (dictionary.centroid(i).bin(j) != 0) {
        is_zero = false;
        break;
      }
    }
    ASSERT_FALSE(is_zero) << "Found zero descriptor.";
  }
}

TEST_F(CodebookTestRealDataNoLocation,
       CodebookBuilderReturnsSameDimensionsAndNoDuplicates) {
  sjm::codebooks::Dictionary dictionary;
  builder_.Cluster(7, 11);
  builder_.GetDictionary(&dictionary);
  for (int a = 0; a < dictionary.centroid_size() - 1; ++a) {
    for (int b = a + 1; b < dictionary.centroid_size(); ++b) {
      float distance_squared = 0;
      ASSERT_EQ(dictionary.centroid(a).bin_size(),
                dictionary.centroid(b).bin_size()) <<
          "Found centroids with mismatching dimensions.";
      for (int i = 0; i < dictionary.centroid(a).bin_size(); ++i) {
        float diff =
            dictionary.centroid(a).bin(i) - dictionary.centroid(b).bin(i);
        distance_squared += diff * diff;
      }
      ASSERT_NE(0, distance_squared) << "Found duplicate descriptors.";
    }
  }
}

TEST_F(CodebookTestRealDataWithLocation,
       CodebookBuilderReturnsSameDimensionsAndNoDuplicates) {
  sjm::codebooks::Dictionary dictionary;
  builder_.Cluster(7, 11);
  builder_.GetDictionary(&dictionary);
  for (int a = 0; a < dictionary.centroid_size() - 1; ++a) {
    for (int b = a + 1; b < dictionary.centroid_size(); ++b) {
      float distance_squared = 0;
      ASSERT_EQ(dictionary.centroid(a).bin_size(),
                dictionary.centroid(b).bin_size()) <<
          "Found centroids with mismatching dimensions.";
      for (int i = 0; i < dictionary.centroid(a).bin_size(); ++i) {
        float diff =
            dictionary.centroid(a).bin(i) - dictionary.centroid(b).bin(i);
        distance_squared += diff * diff;
      }
      ASSERT_NE(0, distance_squared) << "Found duplicate descriptors.";
    }
  }
}

TEST_F(CodebookTestRealDataNoLocation,
       CodebookBuilderReturnsDimensionsThatMatchData) {
  sjm::codebooks::Dictionary dictionary;
  builder_.Cluster(7, 11);
  builder_.GetDictionary(&dictionary);
  for (int i = 0; i < dictionary.centroid_size(); ++i) {
    ASSERT_EQ(data_dimensions_, dictionary.centroid(i).bin_size()) <<
        "Centroid dimensions don't match data dimensions.";
  }
}

TEST_F(CodebookTestRealDataWithLocation,
       CodebookBuilderReturnsDimensionsThatMatchData) {
  sjm::codebooks::Dictionary dictionary;
  builder_.Cluster(7, 11);
  builder_.GetDictionary(&dictionary);
  for (int i = 0; i < dictionary.centroid_size(); ++i) {
    ASSERT_EQ(data_dimensions_, dictionary.centroid(i).bin_size()) <<
        "Centroid dimensions don't match data dimensions.";
  }
}

TEST(CodebookTestDataCount,
     CodebookBuilderReturnsDataCount) {
  sjm::codebooks::CodebookBuilder builder;

  sjm::sift::DescriptorSet generated_descriptor_set;
  for (size_t i = 0; i < 25; ++i) {
    sjm::sift::SiftDescriptor* descriptor =
        generated_descriptor_set.add_sift_descriptor();
    descriptor->add_bin(2);
    descriptor->add_bin(3);
  }

  builder.AddData(generated_descriptor_set, 1.0, 0.0f);
  ASSERT_EQ(25, builder.DataSize());

  generated_descriptor_set.Clear();
  for (size_t i = 0; i < 15; ++i) {
    sjm::sift::SiftDescriptor* descriptor =
        generated_descriptor_set.add_sift_descriptor();
    descriptor->add_bin(2);
    descriptor->add_bin(3);
  }
  builder.AddData(generated_descriptor_set, 1.0, 0.0f);
  ASSERT_EQ(40, builder.DataSize());
}

TEST(CodebookTestDataCount,
     CodebookBuilderAddsPercentageOfData) {
  sjm::codebooks::CodebookBuilder builder;

  sjm::sift::DescriptorSet generated_descriptor_set;
  for (size_t i = 0; i < 10000; ++i) {
    sjm::sift::SiftDescriptor* descriptor =
        generated_descriptor_set.add_sift_descriptor();
    descriptor->add_bin(2);
    descriptor->add_bin(3);
  }

  builder.AddData(generated_descriptor_set, 0.4, 0.0f);
  // Approximately 40% of the data should be used in the
  // builder. That's 4000 descriptors.
  ASSERT_GT(builder.DataSize(), 3800);
  ASSERT_LT(builder.DataSize(), 4200);
}

TEST(CodebookTestGeneratedData,
     CodebookBuilderReturnsGroundTruthCentroidsNoLocation) {
  sjm::codebooks::CodebookBuilder builder;

  vector<pair<float, float> > ground_truth_centroids;
  ground_truth_centroids.push_back(make_pair(26, 15));
  ground_truth_centroids.push_back(make_pair(22, 22));
  ground_truth_centroids.push_back(make_pair(17, 16));

  sjm::sift::DescriptorSet generated_descriptor_set;
  for (size_t i = 0; i < ground_truth_centroids.size(); ++i) {
    float centre_x = ground_truth_centroids[i].first;
    float centre_y = ground_truth_centroids[i].second;
    // Put 100 samples, dispersed uniformly around a disc of radius 5
    // around each ground truth centroid.
    for (int j = 0; j < 10000; ++j) {
      float x_offset = random() / static_cast<float>(RAND_MAX);
      float y_offset = random() / static_cast<float>(RAND_MAX);
      x_offset *= 10;  // Make the radius equal 5.
      y_offset *= 10;
      x_offset -= 5;  // Centre at 0.
      y_offset -= 5;
      sjm::sift::SiftDescriptor* descriptor =
          generated_descriptor_set.add_sift_descriptor();
      // Round these to the nearest integer, because the protocol buffer
      // type is uint32_t.
      descriptor->add_bin(static_cast<int>(centre_x + x_offset + 0.5));
      descriptor->add_bin(static_cast<int>(centre_y + y_offset + 0.5));
    }
  }

  builder.AddData(generated_descriptor_set, 1.0, 0.0f);
  sjm::codebooks::Dictionary dictionary;
  builder.Cluster(3, 11);
  builder.GetDictionary(&dictionary);

  // Check that the dictionary centroids match the ground truth centroids.
  for (size_t i = 0; i < ground_truth_centroids.size(); ++i) {
    bool found = false;
    // Search for this ground truth centroid in the dictionary.
    for (int j = 0; j < dictionary.centroid_size(); ++j) {
      float diff_x = (ground_truth_centroids[i].first -
                      dictionary.centroid(j).bin(0));
      float diff_y = (ground_truth_centroids[i].second -
                      dictionary.centroid(j).bin(1));
      float distance_squared = diff_x * diff_x + diff_y * diff_y;
      if (distance_squared < 0.5) {
        found = true;
        break;
      }
    }
    ASSERT_TRUE(found) << "Ground truth centroid not found in dictionary.";
  }
}

TEST(CodebookTestGeneratedData,
     CodebookBuilderDoesNotReturnGroundTruthCentroidsWithOneIteration) {
  sjm::codebooks::CodebookBuilder builder;

  vector<pair<float, float> > ground_truth_centroids;
  ground_truth_centroids.push_back(make_pair(26, 15));
  ground_truth_centroids.push_back(make_pair(22, 22));
  ground_truth_centroids.push_back(make_pair(17, 16));

  sjm::sift::DescriptorSet generated_descriptor_set;
  for (size_t i = 0; i < ground_truth_centroids.size(); ++i) {
    float centre_x = ground_truth_centroids[i].first;
    float centre_y = ground_truth_centroids[i].second;
    // Put 100 samples, dispersed uniformly around a disc of radius 5
    // around each ground truth centroid.
    for (int j = 0; j < 10000; ++j) {
      float x_offset = random() / static_cast<float>(RAND_MAX);
      float y_offset = random() / static_cast<float>(RAND_MAX);
      x_offset *= 10;  // Make the radius equal 5.
      y_offset *= 10;
      x_offset -= 5;  // Centre at 0.
      y_offset -= 5;
      sjm::sift::SiftDescriptor* descriptor =
          generated_descriptor_set.add_sift_descriptor();
      // Round these to the nearest integer, because the protocol buffer
      // type is uint32_t.
      descriptor->add_bin(static_cast<int>(centre_x + x_offset + 0.5));
      descriptor->add_bin(static_cast<int>(centre_y + y_offset + 0.5));
    }
  }

  builder.AddData(generated_descriptor_set, 1.0, 0.0f);
  sjm::codebooks::Dictionary dictionary;
  builder.Cluster(3, 1);  // 3 centroids, 1 iteration
  builder.GetDictionary(&dictionary);

  bool all_found = true;
  // Check that the dictionary centroids match the ground truth centroids.
  for (size_t i = 0; i < ground_truth_centroids.size(); ++i) {
    bool found = false;
    // Search for this ground truth centroid in the dictionary.
    for (int j = 0; j < dictionary.centroid_size(); ++j) {
      float diff_x = (ground_truth_centroids[i].first -
                      dictionary.centroid(j).bin(0));
      float diff_y = (ground_truth_centroids[i].second -
                      dictionary.centroid(j).bin(1));
      float distance_squared = diff_x * diff_x + diff_y * diff_y;
      if (distance_squared < 0.5) {
        found = true;
        break;
      }
    }
    if (!found) {
      all_found = false;
    }
  }
  ASSERT_FALSE(all_found) << "We should have not found ground truth with only"
      " one iteration.";
}

TEST(CodebookTestGeneratedData,
     CodebookBuilderDoesNotReturnGroundTruthCentroidsWithBadApproximation) {
  sjm::codebooks::CodebookBuilder builder;

  vector<pair<float, float> > ground_truth_centroids;
  for (float angle = 0; angle < 2 * M_PI; angle += M_PI / 64) {
    // The centers are in a circle that's 5 100 units in radius,
    // centered at 105, 105.
    int x = static_cast<int>(std::cos(angle) * 1000 + 1005 + 0.5);
    int y = static_cast<int>(std::sin(angle) * 1000 + 1005 + 0.5);
    ground_truth_centroids.push_back(make_pair(x, y));
  }

  sjm::sift::DescriptorSet generated_descriptor_set;
  for (size_t i = 0; i < ground_truth_centroids.size(); ++i) {
    float centre_x = ground_truth_centroids[i].first;
    float centre_y = ground_truth_centroids[i].second;
    // Put 100 samples, dispersed uniformly around a disc of radius 5
    // around each ground truth centroid.
    for (int j = 0; j < 100; ++j) {
      float x_offset = random() / static_cast<float>(RAND_MAX);
      float y_offset = random() / static_cast<float>(RAND_MAX);
      x_offset *= 6;  // Make the radius equal 3.
      y_offset *= 6;
      x_offset -= 3;  // Centre at 0.
      y_offset -= 3;
      sjm::sift::SiftDescriptor* descriptor =
          generated_descriptor_set.add_sift_descriptor();
      // Round these to the nearest integer, because the protocol buffer
      // type is uint32_t.
      descriptor->add_bin(static_cast<int>(centre_x + x_offset + 0.5));
      descriptor->add_bin(static_cast<int>(centre_y + y_offset + 0.5));
    }
  }

  builder.AddData(generated_descriptor_set, 1.0, 0.0f);
  sjm::codebooks::Dictionary dictionary;
  // 11 iterations, 0.1 accuracy.
  builder.ClusterApproximately(ground_truth_centroids.size(), 11, 0.1,
                               sjm::codebooks::KMEANS_PP);
  builder.GetDictionary(&dictionary);

  int num_found = 0;
  // Check that the dictionary centroids match the ground truth centroids.
  for (size_t i = 0; i < ground_truth_centroids.size(); ++i) {
    // Search for this ground truth centroid in the dictionary.
    for (int j = 0; j < dictionary.centroid_size(); ++j) {
      float diff_x = (ground_truth_centroids[i].first -
                      dictionary.centroid(j).bin(0));
      float diff_y = (ground_truth_centroids[i].second -
                      dictionary.centroid(j).bin(1));
      float distance_squared = diff_x * diff_x + diff_y * diff_y;
      if (distance_squared < 1.0) {
        num_found += 1;
        break;
      }
    }
  }

  LOG(INFO) << static_cast<float>(num_found) / ground_truth_centroids.size() <<
      " of centroids found.";
  ASSERT_LT(static_cast<float>(num_found) / ground_truth_centroids.size(), 0.6);
}

TEST(CodebookTestGeneratedData,
     CodebookBuilderReturnsCloseToGroundTruthCentroidsWithGoodApproximation) {
  sjm::codebooks::CodebookBuilder builder;

  vector<pair<float, float> > ground_truth_centroids;
  for (float angle = 0; angle < 2 * M_PI; angle += M_PI / 64) {
    // The centers are in a circle that's 5 100 units in radius,
    // centered at 105, 105.
    int x = static_cast<int>(std::cos(angle) * 1000 + 1005 + 0.5);
    int y = static_cast<int>(std::sin(angle) * 1000 + 1005 + 0.5);
    ground_truth_centroids.push_back(make_pair(x, y));
  }

  sjm::sift::DescriptorSet generated_descriptor_set;
  for (size_t i = 0; i < ground_truth_centroids.size(); ++i) {
    float centre_x = ground_truth_centroids[i].first;
    float centre_y = ground_truth_centroids[i].second;
    // Put 100 samples, dispersed uniformly around a disc of radius 5
    // around each ground truth centroid.
    for (int j = 0; j < 100; ++j) {
      float x_offset = random() / static_cast<float>(RAND_MAX);
      float y_offset = random() / static_cast<float>(RAND_MAX);
      x_offset *= 6;  // Make the radius equal 3.
      y_offset *= 6;
      x_offset -= 3;  // Centre at 0.
      y_offset -= 3;
      sjm::sift::SiftDescriptor* descriptor =
          generated_descriptor_set.add_sift_descriptor();
      // Round these to the nearest integer, because the protocol buffer
      // type is uint32_t.
      descriptor->add_bin(static_cast<int>(centre_x + x_offset + 0.5));
      descriptor->add_bin(static_cast<int>(centre_y + y_offset + 0.5));
    }
  }

  builder.AddData(generated_descriptor_set, 1.0, 0.0f);
  sjm::codebooks::Dictionary dictionary;
  // 11 iterations, 0.9 accuracy.
  builder.ClusterApproximately(ground_truth_centroids.size(), 11, 0.9,
                               sjm::codebooks::KMEANS_PP);
  builder.GetDictionary(&dictionary);

  int num_found = 0;
  // Check that the dictionary centroids match the ground truth centroids.
  for (size_t i = 0; i < ground_truth_centroids.size(); ++i) {
    // Search for this ground truth centroid in the dictionary.
    for (int j = 0; j < dictionary.centroid_size(); ++j) {
      float diff_x = (ground_truth_centroids[i].first -
                      dictionary.centroid(j).bin(0));
      float diff_y = (ground_truth_centroids[i].second -
                      dictionary.centroid(j).bin(1));
      float distance_squared = diff_x * diff_x + diff_y * diff_y;
      if (distance_squared < 1.0) {
        num_found += 1;
        break;
      }
    }
  }

  LOG(INFO) << static_cast<float>(num_found) / ground_truth_centroids.size() <<
      " of centroids found.";
  ASSERT_GT(static_cast<float>(num_found) / ground_truth_centroids.size(), 0.7);
}

TEST(CodebookTestGeneratedData,
     CodebookBuilderReturnsCloseToGroundTruthCentroidsWithGoodApproximationAndRandomInit) {
  sjm::codebooks::CodebookBuilder builder;

  vector<pair<float, float> > ground_truth_centroids;
  for (float angle = 0; angle < 2 * M_PI; angle += M_PI / 32) {
    // The centers are in a circle that's 1000 units in radius,
    // centered at 1005, 1005.
    int x = static_cast<int>(std::cos(angle) * 1000 + 1005 + 0.5);
    int y = static_cast<int>(std::sin(angle) * 1000 + 1005 + 0.5);
    ground_truth_centroids.push_back(make_pair(x, y));
  }

  sjm::sift::DescriptorSet generated_descriptor_set;
  for (size_t i = 0; i < ground_truth_centroids.size(); ++i) {
    float centre_x = ground_truth_centroids[i].first;
    float centre_y = ground_truth_centroids[i].second;
    // Put 100 samples, dispersed uniformly around a disc of radius 5
    // around each ground truth centroid.
    for (int j = 0; j < 100; ++j) {
      float x_offset = random() / static_cast<float>(RAND_MAX);
      float y_offset = random() / static_cast<float>(RAND_MAX);
      x_offset *= 10;  // Make the radius equal 5.
      y_offset *= 10;
      x_offset -= 5;  // Centre at 0.
      y_offset -= 5;
      sjm::sift::SiftDescriptor* descriptor =
          generated_descriptor_set.add_sift_descriptor();
      // Round these to the nearest integer, because the protocol buffer
      // type is uint32_t.
      descriptor->add_bin(static_cast<int>(centre_x + x_offset + 0.5));
      descriptor->add_bin(static_cast<int>(centre_y + y_offset + 0.5));
    }
  }

  builder.AddData(generated_descriptor_set, 1.0, 0.0f);
  sjm::codebooks::Dictionary dictionary;
  // 81 iterations, 0.9 accuracy, random initialization.
  builder.ClusterApproximately(ground_truth_centroids.size(), 81, 0.9,
                               sjm::codebooks::KMEANS_RANDOM);
  builder.GetDictionary(&dictionary);

  int num_found = 0;
  // Check that the dictionary centroids match the ground truth centroids.
  for (size_t i = 0; i < ground_truth_centroids.size(); ++i) {
    // Search for this ground truth centroid in the dictionary.
    for (int j = 0; j < dictionary.centroid_size(); ++j) {
      float diff_x = (ground_truth_centroids[i].first -
                      dictionary.centroid(j).bin(0));
      float diff_y = (ground_truth_centroids[i].second -
                      dictionary.centroid(j).bin(1));
      float distance_squared = diff_x * diff_x + diff_y * diff_y;
      if (distance_squared < 1.0) {
        num_found += 1;
        break;
      }
    }
  }

  LOG(INFO) << static_cast<float>(num_found) / ground_truth_centroids.size() <<
      " of centroids found.";
  ASSERT_GT(static_cast<float>(num_found) / ground_truth_centroids.size(), 0.3);
}

TEST(CodebookTestGeneratedData,
     CodebookBuilderReturnsGroundTruthCentroidsWithLocation) {
  sjm::codebooks::CodebookBuilder builder;

  vector<pair<float, float> > ground_truth_appearance;
  ground_truth_appearance.push_back(make_pair(26, 15));
  ground_truth_appearance.push_back(make_pair(22, 22));
  ground_truth_appearance.push_back(make_pair(17, 16));
  vector<pair<float, float> > ground_truth_spatial;
  ground_truth_spatial.push_back(make_pair(0.4, 0.2));
  ground_truth_spatial.push_back(make_pair(0.9, 0.1));
  ground_truth_spatial.push_back(make_pair(0.6, 0.8));

  sjm::sift::DescriptorSet generated_descriptor_set;
  for (size_t i = 0; i < ground_truth_appearance.size(); ++i) {
    float centre_d1 = ground_truth_appearance[i].first;
    float centre_d2 = ground_truth_appearance[i].second;
    float centre_x = ground_truth_spatial[i].first;
    float centre_y = ground_truth_spatial[i].second;
    // Put 100 samples, dispersed uniformly around a disc of radius 5
    // around each ground truth centroid.
    for (int j = 0; j < 10000; ++j) {
      float d1_offset = random() / static_cast<float>(RAND_MAX);
      float d2_offset = random() / static_cast<float>(RAND_MAX);
      float x_offset = random() / static_cast<float>(RAND_MAX);
      float y_offset = random() / static_cast<float>(RAND_MAX);
      d1_offset *= 10;  // Make the location radius equal 5.
      d2_offset *= 10;
      d1_offset -= 5;  // Centre at 0.
      d2_offset -= 5;
      x_offset /= 10;  // Make the spatial radius equal 0.1.
      y_offset /= 10;
      x_offset -= 0.05;  // Centre at 0.
      y_offset -= 0.05;
      sjm::sift::SiftDescriptor* descriptor =
          generated_descriptor_set.add_sift_descriptor();
      // Round these to the nearest integer, because the protocol buffer
      // type is uint32_t.
      descriptor->add_bin(static_cast<int>(centre_d1 + d1_offset + 0.5));
      descriptor->add_bin(static_cast<int>(centre_d2 + d2_offset + 0.5));
      descriptor->set_x(centre_x + x_offset);
      descriptor->set_y(centre_y + y_offset);
    }
  }

  const float location_weighting = 1.5;
  builder.AddData(generated_descriptor_set, 1.0, location_weighting);
  sjm::codebooks::Dictionary dictionary;
  builder.Cluster(3, 11);
  builder.GetDictionary(&dictionary);

  // Check that the dictionary centroids match the ground truth centroids.
  for (size_t i = 0; i < ground_truth_appearance.size(); ++i) {
    bool found = false;
    // Search for this ground truth centroid in the dictionary.
    for (int j = 0; j < dictionary.centroid_size(); ++j) {
      float diff_d1 = (ground_truth_appearance[i].first -
                      dictionary.centroid(j).bin(0));
      float diff_d2 = (ground_truth_appearance[i].second -
                      dictionary.centroid(j).bin(1));
      float diff_x = (ground_truth_spatial[i].first * location_weighting * 127 -
                      dictionary.centroid(j).bin(2));
      float diff_y =
          (ground_truth_spatial[i].second * location_weighting * 127 -
           dictionary.centroid(j).bin(3));
      float distance_squared =
          diff_d1 * diff_d1 + diff_d2 * diff_d2 +
          diff_x * diff_x + diff_y * diff_y;
      if (distance_squared < 0.5) {
        found = true;
        break;
      }
    }
    ASSERT_TRUE(found) << "Ground truth centroid not found in dictionary.";
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
