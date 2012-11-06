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

#include "spatial_pyramid/spatial_pyramid_kernel.h"

#include <algorithm>

#include "glog/logging.h"

#include "spatial_pyramid/spatial_pyramid.pb.h"

namespace sjm {
namespace spatial_pyramid {

float HistogramIntersection(const SparseVectorFloat& a,
                            const SparseVectorFloat& b) {
  if (a.value_size() == 0 || b.value_size() == 0) {
    return 0;
  }
  int j = 0;
  float intersection = 0;
  for (int i = 0; i < a.value_size(); ++i) {
    int a_index = a.value(i).index();
    while (b.value(j).index() < a_index &&
           j < b.value_size() - 1) {
      ++j;
    }
    if (a_index == b.value(j).index()) {
      intersection +=
          std::min(a.value(i).value(), b.value(j).value());
    }
  }
  return intersection;
}

float LinearKernel(const SpatialPyramid& pyramid_a,
                   const SpatialPyramid& pyramid_b) {
  CHECK_EQ(pyramid_a.level_size(), pyramid_b.level_size());
  int num_levels = pyramid_a.level_size();

  float dot = 0;
  for (int h = 0; h < pyramid_a.level(0).histogram_size(); ++h) {
    dot += Dot(
        pyramid_a.level(0).histogram(h),
        pyramid_b.level(0).histogram(h));
  }

  for (int level = 1; level < num_levels; ++level) {
    for (int h = 0; h < pyramid_a.level(level).histogram_size(); ++h) {
      dot += Dot(
          pyramid_a.level(level).histogram(h),
          pyramid_b.level(level).histogram(h));
    }
  }
  return dot;
}

float SpmKernel(const SpatialPyramid& pyramid_a,
                const SpatialPyramid& pyramid_b,
                const int num_levels) {
  CHECK_EQ(pyramid_a.level_size(), pyramid_b.level_size());
  CHECK_GE(pyramid_a.level_size(), num_levels);

  int max_level = num_levels - 1;  // Switch to 0-based level ids.
  float intersection = 0;
  for (int h = 0; h < pyramid_a.level(0).histogram_size(); ++h) {
    intersection += HistogramIntersection(
        pyramid_a.level(0).histogram(h),
        pyramid_b.level(0).histogram(h)) / (1 << max_level);
  }

  for (int level = 1; level <= max_level; ++level) {
    for (int h = 0; h < pyramid_a.level(level).histogram_size(); ++h) {
      intersection += HistogramIntersection(
          pyramid_a.level(level).histogram(h),
          pyramid_b.level(level).histogram(h)) / (1 << (max_level - level + 1));
    }
  }
  return intersection;
}

void UnrollHistograms(const SpatialPyramid& pyramid,
                      SparseVectorFloat* result_histogram,
                      int* result_dimensions) {
  result_histogram->Clear();
  CHECK_GT(pyramid.level_size(), 0) << "Pyramid has no levels.";
  int histogram_count = 0;
  int base_index = 0;
  for (int level_id = 0; level_id < pyramid.level_size(); ++level_id) {
    const PyramidLevel& level = pyramid.level(level_id);
    CHECK_EQ(level.rows() * level.columns(),
             level.histogram_size()) << "Number of histograms doesn't " <<
        "match rows * columns.";
    for (int histogram_id = 0; histogram_id < level.histogram_size();
         ++histogram_id) {
      CHECK_NE(-1, level.histogram(histogram_id).non_sparse_length()) <<
          "Can't unroll this spatial pyramid because the non_sparse_length "
          "wasn't recorded.";
      const SparseVectorFloat& histogram = level.histogram(histogram_id);
      for (int i = 0; i < histogram.value_size(); ++i) {
        SparseValueFloat* value = result_histogram->add_value();
        value->set_index(base_index + histogram.value(i).index());
        value->set_value(histogram.value(i).value());
      }
      base_index += level.histogram(histogram_id).non_sparse_length();
    }
  }
  if (result_dimensions != NULL) {
    *result_dimensions = base_index;
  }
}

float Dot(const SparseVectorFloat& a, const SparseVectorFloat& b) {
  float result = 0;
  int j = 0;
  for (int i = 0; i < a.value_size(); ++i) {
    int a_index = a.value(i).index();
    while (j < b.value_size() &&
           b.value(j).index() < a_index) {
      ++j;
    }
    if (j >= b.value_size()) {
      break;
    }
    if (b.value(j).index() == a_index) {
      result += b.value(j).value() * a.value(i).value();
    }
  }
  return result;
}
}}  // namespace.
