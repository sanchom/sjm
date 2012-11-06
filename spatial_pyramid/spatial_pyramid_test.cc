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

// File under test.
#include "spatial_pyramid/spatial_pyramid_builder.h"

#include <vector>

#include "codebooks/dictionary.pb.h"
#include "sift/sift_descriptors.pb.h"
#include "spatial_pyramid/spatial_pyramid.pb.h"
#include "spatial_pyramid/spatial_pyramid_kernel.h"

// Third party includes.
#include "glog/logging.h"
#include "gtest/gtest.h"

class SpatialPyramidTest : public ::testing::Test {
 protected:
  // Returns a simple, two-codeword dictionary. One codeword at (5,6)
  // another at (15,2).
  std::vector<sjm::codebooks::Dictionary> GetTestDictionary() const {
    sjm::codebooks::Dictionary dictionary;
    sjm::codebooks::Centroid* c;
    c = dictionary.add_centroid();
    c->add_bin(5);
    c->add_bin(6);
    c = dictionary.add_centroid();
    c->add_bin(15);
    c->add_bin(2);
    std::vector<sjm::codebooks::Dictionary> dictionaries;
    dictionaries.push_back(dictionary);
    return dictionaries;
  }
};

class SpatialPyramidWithMultipleDictionariesTest : public ::testing::Test {
 protected:
  // Returns two dictionaries in a vector. The first one doesn't
  // include any location information. The second uses location
  // weighting = 1.0.
  std::vector<sjm::codebooks::Dictionary> GetTestDictionaries() const {
    std::vector<sjm::codebooks::Dictionary> dictionaries;
    sjm::codebooks::Dictionary appearance_dictionary;
    sjm::codebooks::Centroid* c;
    c = appearance_dictionary.add_centroid();
    c->add_bin(5);
    c->add_bin(6);
    c = appearance_dictionary.add_centroid();
    c->add_bin(15);
    c->add_bin(2);
    dictionaries.push_back(appearance_dictionary);
    sjm::codebooks::Dictionary spatial_coding_dictionary;
    c = spatial_coding_dictionary.add_centroid();
    c->add_bin(6);
    c->add_bin(8);
    c->add_bin(0.15 * 127);  // x = 19.05
    c->add_bin(0.20 * 127);  // y = 25.40
    c = spatial_coding_dictionary.add_centroid();
    c->add_bin(12);
    c->add_bin(4);
    c->add_bin(0.60 * 127);  // x = 76.2
    c->add_bin(0.65 * 127);  // y = 82.55
    spatial_coding_dictionary.set_location_weighting(1.0);
    dictionaries.push_back(spatial_coding_dictionary);
    return dictionaries;
  }
};

class SpatialPyramidUnrollingTest : public ::testing::Test {
 protected:
  sjm::spatial_pyramid::SpatialPyramid GetEmptyOneLevelPyramid() const {
    sjm::spatial_pyramid::SpatialPyramid pyramid;
    sjm::spatial_pyramid::PyramidLevel* level = NULL;
    sjm::spatial_pyramid::SparseVectorFloat* histogram = NULL;
    level = pyramid.add_level();
    level->set_rows(1);
    level->set_columns(1);
    histogram = level->add_histogram();
    histogram->set_non_sparse_length(10);
    return pyramid;
  }

  sjm::spatial_pyramid::SpatialPyramid GetNonEmptyOneLevelPyramid() const {
    sjm::spatial_pyramid::SpatialPyramid pyramid;
    sjm::spatial_pyramid::PyramidLevel* level = NULL;
    sjm::spatial_pyramid::SparseVectorFloat* histogram = NULL;
    sjm::spatial_pyramid::SparseValueFloat* value = NULL;
    level = pyramid.add_level();
    level->set_rows(1);
    level->set_columns(1);
    histogram = level->add_histogram();
    value = histogram->add_value();
    value->set_index(3);
    value->set_value(0.3);
    value = histogram->add_value();
    value->set_index(5);
    value->set_value(0.4);
    histogram->set_non_sparse_length(10);
    return pyramid;
  }

  sjm::spatial_pyramid::SpatialPyramid GetNonEmptyTwoLevelPyramid() const {
    sjm::spatial_pyramid::SpatialPyramid pyramid;
    sjm::spatial_pyramid::PyramidLevel* level = NULL;
    sjm::spatial_pyramid::SparseVectorFloat* histogram = NULL;
    sjm::spatial_pyramid::SparseValueFloat* value = NULL;
    level = pyramid.add_level();
    level->set_rows(1);
    level->set_columns(1);
    histogram = level->add_histogram();
    histogram->set_non_sparse_length(10);
    value = histogram->add_value();
    value->set_index(8);  // Index 8 in the unrolled version.
    value->set_value(0.4);
    level = pyramid.add_level();
    level->set_rows(2);
    level->set_columns(2);
    histogram = level->add_histogram();
    histogram->set_non_sparse_length(10);
    histogram = level->add_histogram();
    histogram->set_non_sparse_length(10);
    value = histogram->add_value();
    value->set_index(7);  // Index 27 in the unrolled version.
    value->set_value(0.1);
    histogram = level->add_histogram();
    histogram->set_non_sparse_length(10);
    histogram = level->add_histogram();
    histogram->set_non_sparse_length(10);
    return pyramid;
  }
};

class SpatialPyramidWithLocationWeightedDictionaryTest
    : public ::testing::Test {
 protected:
  std::vector<sjm::codebooks::Dictionary> GetTestDictionary(
      const float location_weighting) const {
    sjm::codebooks::Dictionary dictionary;
    sjm::codebooks::Centroid* c;
    c = dictionary.add_centroid();
    c->add_bin(12);
    c->add_bin(20);
    c->add_bin(0.25 * 127 * location_weighting);
    c->add_bin(0.25 * 127 * location_weighting);
    c = dictionary.add_centroid();
    c->add_bin(20);
    c->add_bin(12);
    c->add_bin(0.75 * 127 * location_weighting);
    c->add_bin(0.75 * 127 * location_weighting);
    dictionary.set_location_weighting(location_weighting);
    std::vector<sjm::codebooks::Dictionary> dictionaries;
    dictionaries.push_back(dictionary);
    return dictionaries;
  }
};

TEST(DotTest,
     EmptyDotEmpty) {
  sjm::spatial_pyramid::SparseVectorFloat a;
  sjm::spatial_pyramid::SparseVectorFloat b;
  ASSERT_EQ(0, sjm::spatial_pyramid::Dot(a, b));
}

TEST(DotTest,
     EmptyDotNonEmpty) {
  sjm::spatial_pyramid::SparseVectorFloat a;
  sjm::spatial_pyramid::SparseValueFloat* v = NULL;
  v = a.add_value();
  v->set_index(3);
  v->set_value(0.3);
  sjm::spatial_pyramid::SparseVectorFloat b;
  ASSERT_EQ(0, sjm::spatial_pyramid::Dot(a, b));
}

TEST(DotTest,
     NonEmptyDotEmpty) {
  sjm::spatial_pyramid::SparseVectorFloat a;
  sjm::spatial_pyramid::SparseValueFloat* v = NULL;
  v = a.add_value();
  v->set_index(3);
  v->set_value(0.3);
  sjm::spatial_pyramid::SparseVectorFloat b;
  ASSERT_EQ(0, sjm::spatial_pyramid::Dot(b, a));
}

TEST(DotTest,
     NonEmptyDotNonEmptyNonZero) {
  sjm::spatial_pyramid::SparseVectorFloat a;
  sjm::spatial_pyramid::SparseValueFloat* v = NULL;
  v = a.add_value();
  v->set_index(3);
  v->set_value(0.5);
  sjm::spatial_pyramid::SparseVectorFloat b;
  v = b.add_value();
  v->set_index(3);
  v->set_value(0.1);
  ASSERT_FLOAT_EQ(0.1 * 0.5, sjm::spatial_pyramid::Dot(a, b));
}

TEST(DotTest,
     NonEmptyDotNonEmptyZero) {
  sjm::spatial_pyramid::SparseVectorFloat a;
  sjm::spatial_pyramid::SparseValueFloat* v = NULL;
  v = a.add_value();
  v->set_index(3);
  v->set_value(0.5);
  sjm::spatial_pyramid::SparseVectorFloat b;
  v = b.add_value();
  v->set_index(4);
  v->set_value(0.1);
  ASSERT_FLOAT_EQ(0, sjm::spatial_pyramid::Dot(a, b));
}

TEST(DotTest,
     LongNonEmptyDotLongNonEmptyNonZero) {
  sjm::spatial_pyramid::SparseVectorFloat a;
  sjm::spatial_pyramid::SparseValueFloat* v = NULL;
  v = a.add_value();
  v->set_index(3);
  v->set_value(0.5);
  v = a.add_value();
  v->set_index(6);
  v->set_value(2.0);
  v = a.add_value();
  v->set_index(10);
  v->set_value(5.0);
  sjm::spatial_pyramid::SparseVectorFloat b;
  v = b.add_value();
  v->set_index(4);
  v->set_value(0.1);
  v = b.add_value();
  v->set_index(6);
  v->set_value(1.5);
  v = b.add_value();
  v->set_index(10);
  v->set_value(0.2);
  v = b.add_value();
  v->set_index(12);
  v->set_value(-2);
  ASSERT_FLOAT_EQ(2.0 * 1.5 + 0.2 * 5.0, sjm::spatial_pyramid::Dot(a, b));
}

TEST_F(SpatialPyramidUnrollingTest,
       EmptyOneLevelPyramidUnrolls) {
  sjm::spatial_pyramid::SpatialPyramid pyramid =
      GetEmptyOneLevelPyramid();
  sjm::spatial_pyramid::SparseVectorFloat unrolled_histogram;
  sjm::spatial_pyramid::UnrollHistograms(pyramid, &unrolled_histogram);
  ASSERT_EQ(0, unrolled_histogram.value_size());
}

TEST_F(SpatialPyramidUnrollingTest,
       NonEmptyOneLevelPyramidUnrolls) {
  sjm::spatial_pyramid::SpatialPyramid pyramid =
      GetNonEmptyOneLevelPyramid();
  sjm::spatial_pyramid::SparseVectorFloat unrolled_histogram;
  sjm::spatial_pyramid::UnrollHistograms(pyramid, &unrolled_histogram);
  ASSERT_EQ(2, unrolled_histogram.value_size());
  ASSERT_EQ(3, unrolled_histogram.value(0).index());
  ASSERT_FLOAT_EQ(0.3, unrolled_histogram.value(0).value());
  ASSERT_EQ(5, unrolled_histogram.value(1).index());
  ASSERT_FLOAT_EQ(0.4, unrolled_histogram.value(1).value());
}

TEST_F(SpatialPyramidUnrollingTest,
       NonEmptyTwoLevelPyramidUnrolls) {
  sjm::spatial_pyramid::SpatialPyramid pyramid =
      GetNonEmptyTwoLevelPyramid();
  sjm::spatial_pyramid::SparseVectorFloat unrolled_histogram;
  sjm::spatial_pyramid::UnrollHistograms(pyramid, &unrolled_histogram);
  ASSERT_EQ(2, unrolled_histogram.value_size());
  ASSERT_EQ(8, unrolled_histogram.value(0).index());
  ASSERT_FLOAT_EQ(0.4, unrolled_histogram.value(0).value());
  ASSERT_EQ(27, unrolled_histogram.value(1).index());
  ASSERT_FLOAT_EQ(0.1, unrolled_histogram.value(1).value());
}

TEST_F(SpatialPyramidUnrollingTest,
       LinearKernelIsSameAsUnrollThenDot) {
  sjm::spatial_pyramid::SpatialPyramid pyramid_a =
      GetNonEmptyTwoLevelPyramid();
  sjm::spatial_pyramid::SpatialPyramid pyramid_b =
      GetNonEmptyTwoLevelPyramid();
  sjm::spatial_pyramid::SparseVectorFloat unrolled_histogram_a;
  sjm::spatial_pyramid::UnrollHistograms(pyramid_a, &unrolled_histogram_a);
  sjm::spatial_pyramid::SparseVectorFloat unrolled_histogram_b;
  sjm::spatial_pyramid::UnrollHistograms(pyramid_b, &unrolled_histogram_b);
  float dot = sjm::spatial_pyramid::Dot(unrolled_histogram_a,
                                        unrolled_histogram_b);
  float linear_kernel = sjm::spatial_pyramid::LinearKernel(pyramid_a,
                                                           pyramid_b);
  ASSERT_FLOAT_EQ(dot, linear_kernel);
}

TEST_F(SpatialPyramidUnrollingTest,
       NonEmptyTwoLevelPyramidUnrollsWithDimensions) {
  sjm::spatial_pyramid::SpatialPyramid pyramid =
      GetNonEmptyTwoLevelPyramid();
  sjm::spatial_pyramid::SparseVectorFloat unrolled_histogram;
  int total_dimensions = 0;
  sjm::spatial_pyramid::UnrollHistograms(pyramid, &unrolled_histogram,
                                         &total_dimensions);
  ASSERT_EQ(50, total_dimensions);
}

TEST(SpatialPyramidKernelTest,
     KernelReturnsCorrectValues) {
  sjm::spatial_pyramid::SpatialPyramid pyramid_1;
  sjm::spatial_pyramid::PyramidLevel* level = pyramid_1.add_level();
  level->set_rows(1);
  level->set_columns(1);

  sjm::spatial_pyramid::SparseVectorFloat* histogram = level->add_histogram();
  sjm::spatial_pyramid::SparseValueFloat* bin = histogram->add_value();
  bin->set_index(0);
  bin->set_value(5);
  bin = histogram->add_value();
  bin->set_index(1);
  bin->set_value(3);

  level = pyramid_1.add_level();
  level->set_rows(2);
  level->set_columns(2);

  histogram = level->add_histogram();
  bin = histogram->add_value();
  bin->set_index(0);
  bin->set_value(2);
  bin = histogram->add_value();
  bin->set_index(1);
  bin->set_value(1);

  histogram = level->add_histogram();
  bin = histogram->add_value();
  bin->set_index(0);
  bin->set_value(1);

  histogram = level->add_histogram();
  bin = histogram->add_value();
  bin->set_index(1);
  bin->set_value(2);

  histogram = level->add_histogram();
  bin = histogram->add_value();
  bin->set_index(0);
  bin->set_value(2);

  sjm::spatial_pyramid::SpatialPyramid pyramid_2;
  level = pyramid_2.add_level();
  level->set_rows(1);
  level->set_columns(1);

  histogram = level->add_histogram();
  bin = histogram->add_value();
  bin->set_index(0);
  bin->set_value(2);
  bin = histogram->add_value();
  bin->set_index(1);
  bin->set_value(1);

  level = pyramid_2.add_level();
  level->set_rows(2);
  level->set_columns(2);

  histogram = level->add_histogram();

  histogram = level->add_histogram();
  bin = histogram->add_value();
  bin->set_index(0);
  bin->set_value(1);
  bin = histogram->add_value();
  bin->set_index(1);
  bin->set_value(1);

  histogram = level->add_histogram();

  histogram = level->add_histogram();
  bin = histogram->add_value();
  bin->set_index(0);
  bin->set_value(1);

  // Pyramid 1's levels:
  // {5, 3}
  //
  // {2, 1}, {1, 0}
  // {0, 2}, {2, 0}
  //
  // Pyramid 2's levels:
  // {2, 1}
  //
  // {0, 0}, {1, 1}
  // {0, 0}, {1, 0}

  // One level kernel should == 3: min(5,2) + min(3,1)
  ASSERT_FLOAT_EQ(3, sjm::spatial_pyramid::SpmKernel(pyramid_1, pyramid_2, 1));
  // Two level kernel should == (3/2) + (2/2) = 5/2
  ASSERT_FLOAT_EQ(5.0 / 2.0,
                  sjm::spatial_pyramid::SpmKernel(pyramid_1, pyramid_2, 2));
}

TEST(SpatialPyramidKernelTest,
     KernelReturnsCorrectValuesForSingleLevelGrid) {
  sjm::spatial_pyramid::SpatialPyramid pyramid_1;
  sjm::spatial_pyramid::PyramidLevel* level = pyramid_1.add_level();
  level->set_rows(2);
  level->set_columns(2);

  sjm::spatial_pyramid::SparseVectorFloat* histogram = level->add_histogram();
  sjm::spatial_pyramid::SparseValueFloat* bin = histogram->add_value();
  bin->set_index(0);
  bin->set_value(2);
  bin = histogram->add_value();
  bin->set_index(1);
  bin->set_value(1);

  histogram = level->add_histogram();
  bin = histogram->add_value();
  bin->set_index(0);
  bin->set_value(1);

  histogram = level->add_histogram();
  bin = histogram->add_value();
  bin->set_index(1);
  bin->set_value(2);

  histogram = level->add_histogram();
  bin = histogram->add_value();
  bin->set_index(0);
  bin->set_value(2);

  sjm::spatial_pyramid::SpatialPyramid pyramid_2;
  level = pyramid_2.add_level();
  level->set_rows(2);
  level->set_columns(2);

  histogram = level->add_histogram();

  histogram = level->add_histogram();
  bin = histogram->add_value();
  bin->set_index(0);
  bin->set_value(1);
  bin = histogram->add_value();
  bin->set_index(1);
  bin->set_value(1);

  histogram = level->add_histogram();

  histogram = level->add_histogram();
  bin = histogram->add_value();
  bin->set_index(0);
  bin->set_value(1);

  // Pyramid 1's level:
  // {2, 1}, {1, 0}
  // {0, 2}, {2, 0}
  //
  // Pyramid 2's level:
  // {0, 0}, {1, 1}
  // {0, 0}, {1, 0}

  // One level kernel should == 2
  ASSERT_FLOAT_EQ(2.0,
                  sjm::spatial_pyramid::SpmKernel(pyramid_1, pyramid_2, 1));
}

TEST_F(SpatialPyramidTest,
       GivesRequestedNumberOfLevels) {
  sjm::spatial_pyramid::SpatialPyramidBuilder builder;
  std::vector<sjm::codebooks::Dictionary> dictionary = GetTestDictionary();
  builder.Init(dictionary, 1);
  sjm::sift::DescriptorSet descriptors;
  sjm::spatial_pyramid::SpatialPyramid pyramid;
  builder.BuildPyramid(descriptors, 1, 1, sjm::spatial_pyramid::AVERAGE_POOLING,
                       &pyramid);
  ASSERT_EQ(1, pyramid.level_size());
  ASSERT_EQ(1, pyramid.level(0).histogram_size());
  builder.BuildPyramid(descriptors, 2, 1, sjm::spatial_pyramid::AVERAGE_POOLING,
                       &pyramid);
  ASSERT_EQ(2, pyramid.level_size());
}

TEST_F(SpatialPyramidTest,
       GivesRequestedLevelGeometry) {
  sjm::spatial_pyramid::SpatialPyramidBuilder builder;
  std::vector<sjm::codebooks::Dictionary> dictionary = GetTestDictionary();
  builder.Init(dictionary, 1);
  sjm::sift::DescriptorSet descriptors;
  sjm::spatial_pyramid::SpatialPyramid pyramid;
  builder.BuildPyramid(descriptors, 1, 1, sjm::spatial_pyramid::AVERAGE_POOLING,
                       &pyramid);
  ASSERT_EQ(1, pyramid.level_size());
  ASSERT_EQ(1, pyramid.level(0).rows());
  ASSERT_EQ(1, pyramid.level(0).columns());
  ASSERT_EQ(1, pyramid.level(0).histogram_size());
  builder.BuildPyramid(descriptors, 2, 1, sjm::spatial_pyramid::AVERAGE_POOLING,
                       &pyramid);
  ASSERT_EQ(2, pyramid.level_size());
  ASSERT_EQ(1, pyramid.level(0).rows());
  ASSERT_EQ(1, pyramid.level(0).columns());
  ASSERT_EQ(1, pyramid.level(0).histogram_size());
  ASSERT_EQ(2, pyramid.level(1).rows());
  ASSERT_EQ(2, pyramid.level(1).columns());
  ASSERT_EQ(4, pyramid.level(1).histogram_size());
  builder.BuildPyramid(descriptors, 3, 1, sjm::spatial_pyramid::AVERAGE_POOLING,
                       &pyramid);
  ASSERT_EQ(3, pyramid.level_size());
  ASSERT_EQ(1, pyramid.level(0).rows());
  ASSERT_EQ(1, pyramid.level(0).columns());
  ASSERT_EQ(1, pyramid.level(0).histogram_size());
  ASSERT_EQ(2, pyramid.level(1).rows());
  ASSERT_EQ(2, pyramid.level(1).columns());
  ASSERT_EQ(4, pyramid.level(1).histogram_size());
  ASSERT_EQ(4, pyramid.level(2).rows());
  ASSERT_EQ(4, pyramid.level(2).columns());
  ASSERT_EQ(16, pyramid.level(2).histogram_size());
}

TEST_F(SpatialPyramidTest,
       GivesRequestedSingleLevelGeometry) {
  sjm::spatial_pyramid::SpatialPyramidBuilder builder;
  std::vector<sjm::codebooks::Dictionary> dictionary = GetTestDictionary();
  builder.Init(dictionary, 1);
  sjm::sift::DescriptorSet descriptors;
  sjm::spatial_pyramid::SpatialPyramid pyramid;
  builder.BuildSingleLevel(descriptors, 2, 1,
                           sjm::spatial_pyramid::AVERAGE_POOLING,
                           &pyramid);
  ASSERT_EQ(1, pyramid.level_size());
  ASSERT_EQ(4, pyramid.level(0).rows());
  ASSERT_EQ(4, pyramid.level(0).columns());
  ASSERT_EQ(16, pyramid.level(0).histogram_size());
}

TEST_F(SpatialPyramidTest,
       GivesCorrectBagOfWords) {
  sjm::spatial_pyramid::SpatialPyramidBuilder builder;
  std::vector<sjm::codebooks::Dictionary> dictionary = GetTestDictionary();
  builder.Init(dictionary, 1);

  sjm::sift::DescriptorSet descriptors;
  sjm::sift::SiftDescriptor* d;
  d = descriptors.add_sift_descriptor();
  d->add_bin(4);
  d->add_bin(6);
  d = descriptors.add_sift_descriptor();
  d->add_bin(8);
  d->add_bin(7);
  d = descriptors.add_sift_descriptor();
  d->add_bin(12);
  d->add_bin(0);
  // There should be 2 elements in the first codeword, and 1 in the
  // second before normalization. With average pooling, this turns
  // into {2/3, 1/3}
  sjm::spatial_pyramid::SpatialPyramid pyramid;
  builder.BuildPyramid(descriptors, 1, 1, sjm::spatial_pyramid::AVERAGE_POOLING,
                       &pyramid);

  ASSERT_EQ(1, pyramid.level(0).rows());
  ASSERT_EQ(1, pyramid.level(0).columns());
  ASSERT_EQ(2, pyramid.level(0).histogram(0).value_size());
  ASSERT_EQ(0, pyramid.level(0).histogram(0).value(0).index());
  ASSERT_FLOAT_EQ(2.0 / 3.0, pyramid.level(0).histogram(0).value(0).value());
  ASSERT_EQ(1, pyramid.level(0).histogram(0).value(1).index());
  ASSERT_FLOAT_EQ(1.0 / 3.0, pyramid.level(0).histogram(0).value(1).value());
}

TEST_F(SpatialPyramidWithLocationWeightedDictionaryTest,
       GivesCorrectBagOfWordsWithLowLocationWeighting) {
  sjm::spatial_pyramid::SpatialPyramidBuilder builder;
  std::vector<sjm::codebooks::Dictionary> dictionary = GetTestDictionary(0.01);
  builder.Init(dictionary, 1);

  sjm::sift::DescriptorSet descriptors;
  sjm::sift::SiftDescriptor* d;
  // This descriptor will get assigned to the first codeword.
  d = descriptors.add_sift_descriptor();
  d->add_bin(15);
  d->add_bin(17);
  d->set_x(0.30);
  d->set_y(0.20);
  // This descriptor will also get assigned to the first codeword.
  d = descriptors.add_sift_descriptor();
  d->add_bin(15);
  d->add_bin(17);
  d->set_x(0.70);
  d->set_y(0.80);
  sjm::spatial_pyramid::SpatialPyramid pyramid;
  builder.BuildPyramid(descriptors, 1, 1, sjm::spatial_pyramid::AVERAGE_POOLING,
                       &pyramid);
  ASSERT_EQ(1, pyramid.level(0).histogram(0).value_size());
  ASSERT_EQ(0, pyramid.level(0).histogram(0).value(0).index());
  // All the weight is in the first bin.
  ASSERT_EQ(1, pyramid.level(0).histogram(0).value(0).value());
}

TEST_F(SpatialPyramidWithLocationWeightedDictionaryTest,
       GivesCorrectBagOfWordsWithHighLocationWeighting) {
  sjm::spatial_pyramid::SpatialPyramidBuilder builder;
  std::vector<sjm::codebooks::Dictionary> dictionary = GetTestDictionary(0.5);
  builder.Init(dictionary, 1);

  sjm::sift::DescriptorSet descriptors;
  sjm::sift::SiftDescriptor* d;
  // This descriptor will get assigned to the first codeword.
  d = descriptors.add_sift_descriptor();
  d->add_bin(15);
  d->add_bin(17);
  d->set_x(0.30);
  d->set_y(0.20);
  // This descriptor will also get assigned to the first codeword.
  d = descriptors.add_sift_descriptor();
  d->add_bin(15);
  d->add_bin(17);
  d->set_x(0.70);
  d->set_y(0.80);
  sjm::spatial_pyramid::SpatialPyramid pyramid;
  builder.BuildPyramid(descriptors, 1, 1, sjm::spatial_pyramid::AVERAGE_POOLING,
                       &pyramid);
  ASSERT_EQ(2, pyramid.level(0).histogram(0).value_size());
  // The weight is split evenly between the two bins.
  ASSERT_EQ(0, pyramid.level(0).histogram(0).value(0).index());
  ASSERT_FLOAT_EQ(0.5, pyramid.level(0).histogram(0).value(0).value());
  ASSERT_EQ(1, pyramid.level(0).histogram(0).value(1).index());
  ASSERT_FLOAT_EQ(0.5, pyramid.level(0).histogram(0).value(0).value());
}

TEST_F(SpatialPyramidWithLocationWeightedDictionaryTest,
       GivesCorrectSingleLevelBagOfWordsWithHighLocationWeighting) {
  sjm::spatial_pyramid::SpatialPyramidBuilder builder;
  std::vector<sjm::codebooks::Dictionary> dictionary = GetTestDictionary(0.5);
  builder.Init(dictionary, 1);

  sjm::sift::DescriptorSet descriptors;
  sjm::sift::SiftDescriptor* d;
  // This descriptor will get assigned to the first codeword.
  d = descriptors.add_sift_descriptor();
  d->add_bin(15);
  d->add_bin(17);
  d->set_x(0.30);
  d->set_y(0.20);
  // This descriptor will also get assigned to the first codeword.
  d = descriptors.add_sift_descriptor();
  d->add_bin(15);
  d->add_bin(17);
  d->set_x(0.70);
  d->set_y(0.80);
  sjm::spatial_pyramid::SpatialPyramid pyramid;
  builder.BuildSingleLevel(descriptors, 0, 1,
                           sjm::spatial_pyramid::AVERAGE_POOLING,
                           &pyramid);
  ASSERT_EQ(2, pyramid.level(0).histogram(0).value_size());
  // The weight is split evenly between the two bins.
  ASSERT_EQ(0, pyramid.level(0).histogram(0).value(0).index());
  ASSERT_FLOAT_EQ(0.5, pyramid.level(0).histogram(0).value(0).value());
  ASSERT_EQ(1, pyramid.level(0).histogram(0).value(1).index());
  ASSERT_FLOAT_EQ(0.5, pyramid.level(0).histogram(0).value(0).value());
}

TEST_F(SpatialPyramidTest,
       GivesCorrectSingleLevelBagOfWords) {
  sjm::spatial_pyramid::SpatialPyramidBuilder builder;
  std::vector<sjm::codebooks::Dictionary> dictionary = GetTestDictionary();
  builder.Init(dictionary, 1);

  sjm::sift::DescriptorSet descriptors;
  sjm::sift::SiftDescriptor* d;
  d = descriptors.add_sift_descriptor();
  d->add_bin(4);
  d->add_bin(6);
  d = descriptors.add_sift_descriptor();
  d->add_bin(8);
  d->add_bin(7);
  d = descriptors.add_sift_descriptor();
  d->add_bin(12);
  d->add_bin(0);
  // There should be 2 elements in the first codeword, and 1 in the
  // second before normalization. With average pooling, this turns
  // into {2/3, 1/3}
  sjm::spatial_pyramid::SpatialPyramid pyramid;
  builder.BuildSingleLevel(descriptors, 0, 1,
                           sjm::spatial_pyramid::AVERAGE_POOLING,
                           &pyramid);

  ASSERT_EQ(1, pyramid.level(0).rows());
  ASSERT_EQ(1, pyramid.level(0).columns());
  ASSERT_EQ(2, pyramid.level(0).histogram(0).value_size());
  ASSERT_EQ(0, pyramid.level(0).histogram(0).value(0).index());
  ASSERT_FLOAT_EQ(2.0 / 3.0, pyramid.level(0).histogram(0).value(0).value());
  ASSERT_EQ(1, pyramid.level(0).histogram(0).value(1).index());
  ASSERT_FLOAT_EQ(1.0 / 3.0, pyramid.level(0).histogram(0).value(1).value());
}

TEST_F(SpatialPyramidTest,
       GivesCorrectSpatialPyramidAveragePooling) {
  sjm::spatial_pyramid::SpatialPyramidBuilder builder;
  std::vector<sjm::codebooks::Dictionary> dictionary = GetTestDictionary();
  builder.Init(dictionary, 1);
  // Building a descriptor set that will get vector quantized into
  // codewords 0 and 1 with the following spatial distribution.

  // ---------------------------------
  // |       |   0   |       |       |
  // |  1    |       |       |       |
  // |       |       |       |  0    |
  // ---------------------------------
  // |       |       |       |       |
  // |       |0      |   0   |    0  |
  // |       |       |   1   |       |
  // ---------------------------------
  // |       | 1     |       |       |
  // |  0    |       |       |  1    |
  // |       |       |       |       |
  // ---------------------------------
  // |       |       |       |       |
  // |       |       | 1     |       |
  // |       | 0     |       |       |
  // ---------------------------------

  // Level 0 Histogram (-> average_pooled)
  // {7,5} -> {7/12, 5/12}

  // Level 1 Histograms (-> average_pooled)
  // {2, 1}, {3, 1} -> {2/3, 1/3}, {3/4, 1/4}
  // {2, 1}, {0, 2} -> {2/3, 1/3}, {0, 1}

  // Level 2 Histograms (-> average_pooled)
  // {0, 1}, {0, 1}, {0, 0}, {1, 0} -> {0, 1}, {0, 1}, {0, 0}, {1, 0}
  // {0, 0}, {1, 0}, {1, 1}, {1, 0} -> {0, 0}, {1, 0}, {1/2, 1/2}, {1, 0}
  // {1, 0}, {0, 1}, {0, 0}, {0, 1} -> {1, 0}, {0, 1}, {0, 0}, {0, 1}
  // {0, 0}, {1, 0}, {0, 1}, {0, 0} -> {0, 0}, {1, 0}, {0, 1}, {0, 0}

  sjm::sift::DescriptorSet descriptors;
  sjm::sift::SiftDescriptor* d;
  // Codeword 1 in row 0, column 0
  d = descriptors.add_sift_descriptor();
  d->set_x(0.125);
  d->set_y(0.125);
  d->add_bin(15);
  d->add_bin(1);
  // Codeword 0 in row 0, column 1
  d = descriptors.add_sift_descriptor();
  d->set_x(0.3);
  d->set_y(0.15);
  d->add_bin(4);
  d->add_bin(7);
  // Codeword 0 in row 0, column 3
  d = descriptors.add_sift_descriptor();
  d->set_x(0.94);
  d->set_y(0.23);
  d->add_bin(7);
  d->add_bin(3);
  // Codeword 0 in row 1, column 1
  d = descriptors.add_sift_descriptor();
  d->set_x(0.33);
  d->set_y(0.43);
  d->add_bin(4);
  d->add_bin(9);
  // Codeword 0 in row 1, column 2
  d = descriptors.add_sift_descriptor();
  d->set_x(0.54);
  d->set_y(0.35);
  d->add_bin(5);
  d->add_bin(6);
  // Codeword 1 in row 1, column 2
  d = descriptors.add_sift_descriptor();
  d->set_x(0.54);
  d->set_y(0.45);
  d->add_bin(17);
  d->add_bin(6);
  // Codeword 0 in row 1, column 3
  d = descriptors.add_sift_descriptor();
  d->set_x(0.85);
  d->set_y(0.40);
  d->add_bin(9);
  d->add_bin(5);
  // Codeword 0 in row 2, column 0
  d = descriptors.add_sift_descriptor();
  d->set_x(0.12);
  d->set_y(0.64);
  d->add_bin(5);
  d->add_bin(1);
  // Codeword 1 in row 2, column 1
  d = descriptors.add_sift_descriptor();
  d->set_x(0.28);
  d->set_y(0.52);
  d->add_bin(20);
  d->add_bin(7);
  // Codeword 1 in row 2, column 3
  d = descriptors.add_sift_descriptor();
  d->set_x(0.80);
  d->set_y(0.60);
  d->add_bin(17);
  d->add_bin(2);
  // Codeword 0 in row 3, column 1
  d = descriptors.add_sift_descriptor();
  d->set_x(0.30);
  d->set_y(0.95);
  d->add_bin(4);
  d->add_bin(6);
  // Codeword 1 in row 3, column 2
  d = descriptors.add_sift_descriptor();
  d->set_x(0.56);
  d->set_y(0.80);
  d->add_bin(15);
  d->add_bin(3);

  sjm::spatial_pyramid::SpatialPyramid pyramid;
  builder.BuildPyramid(descriptors, 3, 1, sjm::spatial_pyramid::AVERAGE_POOLING,
                       &pyramid);

  // Check geometry.
  ASSERT_EQ(3, pyramid.level_size());
  ASSERT_EQ(1, pyramid.level(0).rows());
  ASSERT_EQ(1, pyramid.level(0).columns());
  ASSERT_EQ(1, pyramid.level(0).histogram_size());
  ASSERT_EQ(2, pyramid.level(1).rows());
  ASSERT_EQ(2, pyramid.level(1).columns());
  ASSERT_EQ(4, pyramid.level(1).histogram_size());
  ASSERT_EQ(4, pyramid.level(2).rows());
  ASSERT_EQ(4, pyramid.level(2).columns());
  ASSERT_EQ(16, pyramid.level(2).histogram_size());

  // Level 0 Histogram:
  // {7/12, 5/12}
  ASSERT_EQ(2, pyramid.level(0).histogram(0).value_size());
  ASSERT_EQ(0, pyramid.level(0).histogram(0).value(0).index());
  ASSERT_FLOAT_EQ(7.0 / 12.0, pyramid.level(0).histogram(0).value(0).value());
  ASSERT_EQ(1, pyramid.level(0).histogram(0).value(1).index());
  ASSERT_FLOAT_EQ(5.0 / 12.0, pyramid.level(0).histogram(0).value(1).value());

  // Level 1 Histograms:
  // {2/3, 1/3}, {3/4, 1/4}
  // {2/3, 1/3}, {0, 1}
  ASSERT_EQ(2, pyramid.level(1).histogram(0).value_size());
  ASSERT_EQ(0, pyramid.level(1).histogram(0).value(0).index());
  ASSERT_FLOAT_EQ(2.0 / 3.0, pyramid.level(1).histogram(0).value(0).value());
  ASSERT_EQ(1, pyramid.level(1).histogram(0).value(1).index());
  ASSERT_FLOAT_EQ(1.0 / 3.0, pyramid.level(1).histogram(0).value(1).value());

  ASSERT_EQ(2, pyramid.level(1).histogram(1).value_size());
  ASSERT_EQ(0, pyramid.level(1).histogram(1).value(0).index());
  ASSERT_FLOAT_EQ(3.0 / 4.0, pyramid.level(1).histogram(1).value(0).value());
  ASSERT_EQ(1, pyramid.level(1).histogram(1).value(1).index());
  ASSERT_FLOAT_EQ(1.0 / 4.0, pyramid.level(1).histogram(1).value(1).value());

  ASSERT_EQ(2, pyramid.level(1).histogram(2).value_size());
  ASSERT_EQ(0, pyramid.level(1).histogram(2).value(0).index());
  ASSERT_FLOAT_EQ(2.0 / 3.0, pyramid.level(1).histogram(2).value(0).value());
  ASSERT_EQ(1, pyramid.level(1).histogram(2).value(1).index());
  ASSERT_FLOAT_EQ(1.0 / 3.0, pyramid.level(1).histogram(2).value(1).value());

  ASSERT_EQ(1, pyramid.level(1).histogram(3).value_size());
  ASSERT_EQ(1, pyramid.level(1).histogram(3).value(0).index());
  ASSERT_FLOAT_EQ(1.0, pyramid.level(1).histogram(3).value(0).value());
}

TEST_F(SpatialPyramidTest,
       GivesCorrectSingleLevelAveragePooling) {
  sjm::spatial_pyramid::SpatialPyramidBuilder builder;
  std::vector<sjm::codebooks::Dictionary> dictionary = GetTestDictionary();
  builder.Init(dictionary, 1);
  // Building a descriptor set that will get vector quantized into
  // codewords 0 and 1 with the following spatial distribution.

  // ---------------------------------
  // |       |   0   |       |       |
  // |  1    |       |       |       |
  // |       |       |       |  0    |
  // ---------------------------------
  // |       |       |       |       |
  // |       |0      |   0   |    0  |
  // |       |       |   1   |       |
  // ---------------------------------
  // |       | 1     |       |       |
  // |  0    |       |       |  1    |
  // |       |       |       |       |
  // ---------------------------------
  // |       |       |       |       |
  // |       |       | 1     |       |
  // |       | 0     |       |       |
  // ---------------------------------

  // Level 1 Histograms (-> average_pooled)
  // {2, 1}, {3, 1} -> {2/3, 1/3}, {3/4, 1/4}
  // {2, 1}, {0, 2} -> {2/3, 1/3}, {0, 1}

  sjm::sift::DescriptorSet descriptors;
  sjm::sift::SiftDescriptor* d;
  // Codeword 1 in row 0, column 0
  d = descriptors.add_sift_descriptor();
  d->set_x(0.125);
  d->set_y(0.125);
  d->add_bin(15);
  d->add_bin(1);
  // Codeword 0 in row 0, column 1
  d = descriptors.add_sift_descriptor();
  d->set_x(0.3);
  d->set_y(0.15);
  d->add_bin(4);
  d->add_bin(7);
  // Codeword 0 in row 0, column 3
  d = descriptors.add_sift_descriptor();
  d->set_x(0.94);
  d->set_y(0.23);
  d->add_bin(7);
  d->add_bin(3);
  // Codeword 0 in row 1, column 1
  d = descriptors.add_sift_descriptor();
  d->set_x(0.33);
  d->set_y(0.43);
  d->add_bin(4);
  d->add_bin(9);
  // Codeword 0 in row 1, column 2
  d = descriptors.add_sift_descriptor();
  d->set_x(0.54);
  d->set_y(0.35);
  d->add_bin(5);
  d->add_bin(6);
  // Codeword 1 in row 1, column 2
  d = descriptors.add_sift_descriptor();
  d->set_x(0.54);
  d->set_y(0.45);
  d->add_bin(17);
  d->add_bin(6);
  // Codeword 0 in row 1, column 3
  d = descriptors.add_sift_descriptor();
  d->set_x(0.85);
  d->set_y(0.40);
  d->add_bin(9);
  d->add_bin(5);
  // Codeword 0 in row 2, column 0
  d = descriptors.add_sift_descriptor();
  d->set_x(0.12);
  d->set_y(0.64);
  d->add_bin(5);
  d->add_bin(1);
  // Codeword 1 in row 2, column 1
  d = descriptors.add_sift_descriptor();
  d->set_x(0.28);
  d->set_y(0.52);
  d->add_bin(20);
  d->add_bin(7);
  // Codeword 1 in row 2, column 3
  d = descriptors.add_sift_descriptor();
  d->set_x(0.80);
  d->set_y(0.60);
  d->add_bin(17);
  d->add_bin(2);
  // Codeword 0 in row 3, column 1
  d = descriptors.add_sift_descriptor();
  d->set_x(0.30);
  d->set_y(0.95);
  d->add_bin(4);
  d->add_bin(6);
  // Codeword 1 in row 3, column 2
  d = descriptors.add_sift_descriptor();
  d->set_x(0.56);
  d->set_y(0.80);
  d->add_bin(15);
  d->add_bin(3);

  sjm::spatial_pyramid::SpatialPyramid pyramid;
  builder.BuildSingleLevel(descriptors, 1, 1,
                           sjm::spatial_pyramid::AVERAGE_POOLING,
                           &pyramid);

  // Check geometry.
  ASSERT_EQ(1, pyramid.level_size());
  ASSERT_EQ(2, pyramid.level(0).rows());
  ASSERT_EQ(2, pyramid.level(0).columns());
  ASSERT_EQ(4, pyramid.level(0).histogram_size());

  // Level 1 Histograms:
  // {2/3, 1/3}, {3/4, 1/4}
  // {2/3, 1/3}, {0, 1}
  ASSERT_EQ(2, pyramid.level(0).histogram(0).value_size());
  ASSERT_EQ(0, pyramid.level(0).histogram(0).value(0).index());
  ASSERT_FLOAT_EQ(2.0 / 3.0, pyramid.level(0).histogram(0).value(0).value());
  ASSERT_EQ(1, pyramid.level(0).histogram(0).value(1).index());
  ASSERT_FLOAT_EQ(1.0 / 3.0, pyramid.level(0).histogram(0).value(1).value());

  ASSERT_EQ(2, pyramid.level(0).histogram(1).value_size());
  ASSERT_EQ(0, pyramid.level(0).histogram(1).value(0).index());
  ASSERT_FLOAT_EQ(3.0 / 4.0, pyramid.level(0).histogram(1).value(0).value());
  ASSERT_EQ(1, pyramid.level(0).histogram(1).value(1).index());
  ASSERT_FLOAT_EQ(1.0 / 4.0, pyramid.level(0).histogram(1).value(1).value());

  ASSERT_EQ(2, pyramid.level(0).histogram(2).value_size());
  ASSERT_EQ(0, pyramid.level(0).histogram(2).value(0).index());
  ASSERT_FLOAT_EQ(2.0 / 3.0, pyramid.level(0).histogram(2).value(0).value());
  ASSERT_EQ(1, pyramid.level(0).histogram(2).value(1).index());
  ASSERT_FLOAT_EQ(1.0 / 3.0, pyramid.level(0).histogram(2).value(1).value());

  ASSERT_EQ(1, pyramid.level(0).histogram(3).value_size());
  ASSERT_EQ(1, pyramid.level(0).histogram(3).value(0).index());
  ASSERT_FLOAT_EQ(1.0, pyramid.level(0).histogram(3).value(0).value());
}

TEST_F(SpatialPyramidTest,
       GivesCorrectSpatialPyramidMaxPooling) {
  sjm::spatial_pyramid::SpatialPyramidBuilder builder;
  std::vector<sjm::codebooks::Dictionary> dictionary = GetTestDictionary();
  builder.Init(dictionary, 1);
  // Building a descriptor set that will get vector quantized into
  // codewords 0 and 1 with the following spatial distribution.

  // ---------------------------------
  // |       |   0   |       |       |
  // |  1    |       |       |       |
  // |       |       |       |  0    |
  // ---------------------------------
  // |       |       |       |       |
  // |       |0      |   0   |    0  |
  // |       |       |   1   |       |
  // ---------------------------------
  // |       | 1     |       |       |
  // |  0    |       |       |  1    |
  // |       |       |       |       |
  // ---------------------------------
  // |       |       |       |       |
  // |       |       | 1     |       |
  // |       | 0     |       |       |
  // ---------------------------------

  // Level 0 Histogram (-> max_pooled)
  // {7,5} -> {1, 1}

  // Level 1 Histograms (-> max_pooled)
  // {2, 1}, {3, 1} -> {1, 1}, {1, 1}
  // {2, 1}, {0, 2} -> {1, 1}, {0, 1}

  // Level 2 Histograms (-> max_pooled)
  // {0, 1}, {0, 1}, {0, 0}, {1, 0} -> {0, 1}, {0, 1}, {0, 0}, {1, 0}
  // {0, 0}, {1, 0}, {1, 1}, {1, 0} -> {0, 0}, {1, 0}, {1, 1}, {1, 0}
  // {1, 0}, {0, 1}, {0, 0}, {0, 1} -> {1, 0}, {0, 1}, {0, 0}, {0, 1}
  // {0, 0}, {1, 0}, {0, 1}, {0, 0} -> {0, 0}, {1, 0}, {0, 1}, {0, 0}

  sjm::sift::DescriptorSet descriptors;
  sjm::sift::SiftDescriptor* d;
  // Codeword 1 in row 0, column 0
  d = descriptors.add_sift_descriptor();
  d->set_x(0.125);
  d->set_y(0.125);
  d->add_bin(15);
  d->add_bin(1);
  // Codeword 0 in row 0, column 1
  d = descriptors.add_sift_descriptor();
  d->set_x(0.3);
  d->set_y(0.15);
  d->add_bin(4);
  d->add_bin(7);
  // Codeword 0 in row 0, column 3
  d = descriptors.add_sift_descriptor();
  d->set_x(0.94);
  d->set_y(0.23);
  d->add_bin(7);
  d->add_bin(3);
  // Codeword 0 in row 1, column 1
  d = descriptors.add_sift_descriptor();
  d->set_x(0.33);
  d->set_y(0.43);
  d->add_bin(4);
  d->add_bin(9);
  // Codeword 0 in row 1, column 2
  d = descriptors.add_sift_descriptor();
  d->set_x(0.54);
  d->set_y(0.35);
  d->add_bin(5);
  d->add_bin(6);
  // Codeword 1 in row 1, column 2
  d = descriptors.add_sift_descriptor();
  d->set_x(0.54);
  d->set_y(0.45);
  d->add_bin(17);
  d->add_bin(6);
  // Codeword 0 in row 1, column 3
  d = descriptors.add_sift_descriptor();
  d->set_x(0.85);
  d->set_y(0.40);
  d->add_bin(9);
  d->add_bin(5);
  // Codeword 0 in row 2, column 0
  d = descriptors.add_sift_descriptor();
  d->set_x(0.12);
  d->set_y(0.64);
  d->add_bin(5);
  d->add_bin(1);
  // Codeword 1 in row 2, column 1
  d = descriptors.add_sift_descriptor();
  d->set_x(0.28);
  d->set_y(0.52);
  d->add_bin(20);
  d->add_bin(7);
  // Codeword 1 in row 2, column 3
  d = descriptors.add_sift_descriptor();
  d->set_x(0.80);
  d->set_y(0.60);
  d->add_bin(17);
  d->add_bin(2);
  // Codeword 0 in row 3, column 1
  d = descriptors.add_sift_descriptor();
  d->set_x(0.30);
  d->set_y(0.95);
  d->add_bin(4);
  d->add_bin(6);
  // Codeword 1 in row 3, column 2
  d = descriptors.add_sift_descriptor();
  d->set_x(0.56);
  d->set_y(0.80);
  d->add_bin(15);
  d->add_bin(3);

  sjm::spatial_pyramid::SpatialPyramid pyramid;
  builder.BuildPyramid(descriptors, 3, 1, sjm::spatial_pyramid::MAX_POOLING,
                       &pyramid);

  // Check geometry.
  ASSERT_EQ(3, pyramid.level_size());
  ASSERT_EQ(1, pyramid.level(0).rows());
  ASSERT_EQ(1, pyramid.level(0).columns());
  ASSERT_EQ(1, pyramid.level(0).histogram_size());
  ASSERT_EQ(2, pyramid.level(1).rows());
  ASSERT_EQ(2, pyramid.level(1).columns());
  ASSERT_EQ(4, pyramid.level(1).histogram_size());
  ASSERT_EQ(4, pyramid.level(2).rows());
  ASSERT_EQ(4, pyramid.level(2).columns());
  ASSERT_EQ(16, pyramid.level(2).histogram_size());

  // Level 0 Histogram:
  // {1, 1}
  ASSERT_EQ(2, pyramid.level(0).histogram(0).value_size());
  ASSERT_EQ(0, pyramid.level(0).histogram(0).value(0).index());
  ASSERT_EQ(1, pyramid.level(0).histogram(0).value(0).value());
  ASSERT_EQ(1, pyramid.level(0).histogram(0).value(1).index());
  ASSERT_EQ(1, pyramid.level(0).histogram(0).value(1).value());

  // Level 1 Histograms:
  // {1, 1}, {1, 1}
  // {1 ,1}, {0, 1}
  ASSERT_EQ(2, pyramid.level(1).histogram(0).value_size());
  ASSERT_EQ(0, pyramid.level(1).histogram(0).value(0).index());
  ASSERT_EQ(1, pyramid.level(1).histogram(0).value(0).value());
  ASSERT_EQ(1, pyramid.level(1).histogram(0).value(1).index());
  ASSERT_EQ(1, pyramid.level(1).histogram(0).value(1).value());

  ASSERT_EQ(2, pyramid.level(1).histogram(1).value_size());
  ASSERT_EQ(0, pyramid.level(1).histogram(1).value(0).index());
  ASSERT_EQ(1, pyramid.level(1).histogram(1).value(0).value());
  ASSERT_EQ(1, pyramid.level(1).histogram(1).value(1).index());
  ASSERT_EQ(1, pyramid.level(1).histogram(1).value(1).value());

  ASSERT_EQ(2, pyramid.level(1).histogram(2).value_size());
  ASSERT_EQ(0, pyramid.level(1).histogram(2).value(0).index());
  ASSERT_EQ(1, pyramid.level(1).histogram(2).value(0).value());
  ASSERT_EQ(1, pyramid.level(1).histogram(2).value(1).index());
  ASSERT_EQ(1, pyramid.level(1).histogram(2).value(1).value());

  ASSERT_EQ(1, pyramid.level(1).histogram(3).value_size());
  ASSERT_EQ(1, pyramid.level(1).histogram(3).value(0).index());
  ASSERT_EQ(1, pyramid.level(1).histogram(3).value(0).value());
}

TEST_F(SpatialPyramidTest,
       DoesSoftAssignment) {
  sjm::spatial_pyramid::SpatialPyramidBuilder builder;
  std::vector<sjm::codebooks::Dictionary> dictionary = GetTestDictionary();
  builder.Init(dictionary, 1);

  sjm::sift::DescriptorSet descriptors;
  sjm::sift::SiftDescriptor* d;
  // Closer to codeword 1
  d = descriptors.add_sift_descriptor();
  d->set_x(0.125);  // This won't matter for this test.
  d->set_y(0.125);  // This won't matter for this test.
  d->add_bin(15);
  d->add_bin(1);

  // Just a one-level pyramid. A bag-of-words.
  sjm::spatial_pyramid::SpatialPyramid pyramid;
  builder.BuildPyramid(descriptors, 1, 2, sjm::spatial_pyramid::AVERAGE_POOLING,
                       &pyramid);

  // The result should have two codewords activated (we've lost
  // sparsity because of soft assignment), and codeword 1 should have
  // a higher activation than codeword 0.

  // Checking 2 activations.
  ASSERT_EQ(2, pyramid.level(0).histogram(0).value_size());
  // Check that they're codewords 0 and 1.
  ASSERT_EQ(0, pyramid.level(0).histogram(0).value(0).index());
  ASSERT_EQ(1, pyramid.level(0).histogram(0).value(1).index());
  // Check that codeword 1 has more weight than codeword 0, and both
  // are between 0 and 1.
  float weight_0 = pyramid.level(0).histogram(0).value(0).value();
  float weight_1 = pyramid.level(0).histogram(0).value(1).value();
  ASSERT_GT(weight_1, weight_0);
  ASSERT_GT(weight_0, 0);
  ASSERT_LT(weight_0, 1);
  ASSERT_GT(weight_1, 0);
  ASSERT_LT(weight_1, 1);
}

TEST_F(SpatialPyramidTest,
       SoftAssignmentIsCappedAtDictionarySize) {
  sjm::spatial_pyramid::SpatialPyramidBuilder builder;
  std::vector<sjm::codebooks::Dictionary> dictionary = GetTestDictionary();
  builder.Init(dictionary, 1);

  sjm::sift::DescriptorSet descriptors;
  sjm::sift::SiftDescriptor* d;
  // Closer to codeword 1
  d = descriptors.add_sift_descriptor();
  d->set_x(0.125);  // This won't matter for this test.
  d->set_y(0.125);  // This won't matter for this test.
  d->add_bin(15);
  d->add_bin(1);

  // Just a one-level pyramid. A bag-of-words.
  sjm::spatial_pyramid::SpatialPyramid pyramid;
  // Note: We're requesting a k larger than the size of the
  // dictionary. This should give the same results as if we'd
  // requested k = 2.
  builder.BuildPyramid(descriptors, 1, 5, sjm::spatial_pyramid::AVERAGE_POOLING,
                       &pyramid);

  // The result should have two codewords activated (we've lost
  // sparsity because of soft assignment), and codeword 1 should have
  // a higher activation than codeword 0.

  // Checking 2 activations.
  ASSERT_EQ(2, pyramid.level(0).histogram(0).value_size());
  // Check that they're codewords 0 and 1.
  ASSERT_EQ(0, pyramid.level(0).histogram(0).value(0).index());
  ASSERT_EQ(1, pyramid.level(0).histogram(0).value(1).index());
  // Check that codeword 1 has more weight than codeword 0, and both
  // are between 0 and 1.
  float weight_0 = pyramid.level(0).histogram(0).value(0).value();
  float weight_1 = pyramid.level(0).histogram(0).value(1).value();
  ASSERT_GT(weight_1, weight_0);
  ASSERT_GT(weight_0, 0);
  ASSERT_LT(weight_0, 1);
  ASSERT_GT(weight_1, 0);
  ASSERT_LT(weight_1, 1);
}

TEST_F(SpatialPyramidWithMultipleDictionariesTest,
       GivesCorrectHardCodedOneLevelHistogramWithSingleThread) {
  sjm::spatial_pyramid::SpatialPyramidBuilder builder;
  std::vector<sjm::codebooks::Dictionary> dictionaries = GetTestDictionaries();
  builder.Init(dictionaries, 1);

  // Create descriptor set with locations that matter for the test
  // dictionaries.
  sjm::sift::DescriptorSet descriptors;
  sjm::sift::SiftDescriptor* d;
  // This descriptor is closer to bin 1 (0-based) in the first
  // dictionary and closer to bin 0 in the second histogram.
  d = descriptors.add_sift_descriptor();
  d->set_x(0.15);  // 19.05
  d->set_y(0.20);  // 25.40
  d->add_bin(15);
  d->add_bin(1);
  // This descriptor is closer to bin 0 in the first dictionary and
  // closer to bin 0 in the second histogram.
  d = descriptors.add_sift_descriptor();
  d->set_x(0.15);
  d->set_y(0.20);
  d->add_bin(3);
  d->add_bin(9);

  // The resulting, concatenated histogram should be:
  //
  // [0.5, 0.5, 1.0, 0]

  // Just a one-level pyramid. A bag-of-words, but using concatenated
  // histograms from the two dictionaries.
  sjm::spatial_pyramid::SpatialPyramid pyramid;
  builder.BuildPyramid(descriptors, 1, 1, sjm::spatial_pyramid::AVERAGE_POOLING,
                       &pyramid);

  ASSERT_EQ(1, pyramid.level_size());
  ASSERT_EQ(1, pyramid.level(0).histogram_size());
  ASSERT_EQ(4, pyramid.level(0).histogram(0).non_sparse_length());
  ASSERT_EQ(3, pyramid.level(0).histogram(0).value_size());
  ASSERT_EQ(0, pyramid.level(0).histogram(0).value(0).index());
  ASSERT_FLOAT_EQ(0.5, pyramid.level(0).histogram(0).value(0).value());
  ASSERT_EQ(1, pyramid.level(0).histogram(0).value(1).index());
  ASSERT_FLOAT_EQ(0.5, pyramid.level(0).histogram(0).value(1).value());
  ASSERT_EQ(2, pyramid.level(0).histogram(0).value(2).index());
  ASSERT_FLOAT_EQ(1, pyramid.level(0).histogram(0).value(2).value());
}

TEST_F(SpatialPyramidWithMultipleDictionariesTest,
       GivesCorrectHardCodedOneLevelHistogramWithMultipleThreads) {
  sjm::spatial_pyramid::SpatialPyramidBuilder builder;
  std::vector<sjm::codebooks::Dictionary> dictionaries = GetTestDictionaries();
  // Using up to three threads.
  builder.Init(dictionaries, 3);

  // Create descriptor set with locations that matter for the test
  // dictionaries.
  sjm::sift::DescriptorSet descriptors;
  sjm::sift::SiftDescriptor* d;
  // This descriptor is closer to bin 1 (0-based) in the first
  // dictionary and closer to bin 0 in the second histogram.
  d = descriptors.add_sift_descriptor();
  d->set_x(0.15);  // 19.05
  d->set_y(0.20);  // 25.40
  d->add_bin(15);
  d->add_bin(1);
  // This descriptor is closer to bin 0 in the first dictionary and
  // closer to bin 0 in the second histogram.
  d = descriptors.add_sift_descriptor();
  d->set_x(0.15);
  d->set_y(0.20);
  d->add_bin(3);
  d->add_bin(9);

  // The resulting, concatenated histogram should be:
  //
  // [0.5, 0.5, 1.0, 0]

  // Just a one-level pyramid. A bag-of-words, but using concatenated
  // histograms from the two dictionaries.
  sjm::spatial_pyramid::SpatialPyramid pyramid;
  builder.BuildPyramid(descriptors, 1, 1, sjm::spatial_pyramid::AVERAGE_POOLING,
                       &pyramid);

  ASSERT_EQ(1, pyramid.level_size());
  ASSERT_EQ(1, pyramid.level(0).histogram_size());
  ASSERT_EQ(4, pyramid.level(0).histogram(0).non_sparse_length());
  ASSERT_EQ(3, pyramid.level(0).histogram(0).value_size());
  ASSERT_EQ(0, pyramid.level(0).histogram(0).value(0).index());
  ASSERT_FLOAT_EQ(0.5, pyramid.level(0).histogram(0).value(0).value());
  ASSERT_EQ(1, pyramid.level(0).histogram(0).value(1).index());
  ASSERT_FLOAT_EQ(0.5, pyramid.level(0).histogram(0).value(1).value());
  ASSERT_EQ(2, pyramid.level(0).histogram(0).value(2).index());
  ASSERT_FLOAT_EQ(1, pyramid.level(0).histogram(0).value(2).value());
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
