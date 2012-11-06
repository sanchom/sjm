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

#include "spatial_pyramid/spatial_pyramid_builder.h"

#include <algorithm>
#include <map>
#include <vector>

#include "boost/thread.hpp"
#include "flann/flann.hpp"
#include "glog/logging.h"

#include "codebooks/dictionary.pb.h"
#include "sift/sift_descriptors.pb.h"
#include "spatial_pyramid/spatial_pyramid.pb.h"
#include "util/util.h"

using std::map;

namespace sjm {
namespace spatial_pyramid {

bool SpatialPyramidBuilder::Init(
    const std::vector<sjm::codebooks::Dictionary>& dictionaries,
    const int num_threads) {
  num_threads_ = num_threads;
  if (dictionaries.size() == 0) {
    // We need at least one dictionary in the vector.
    return false;
  }

  // Delete any previous data/indices.
  for (size_t i = 0; i < dictionary_data_.size(); ++i) {
    if (dictionary_data_[i]) {
      delete[] dictionary_data_[i]->ptr();
      delete dictionary_data_[i];
    }
  }
  for (size_t i = 0; i < dictionary_indices_.size(); ++i) {
    if (dictionary_indices_[i]) {
      delete dictionary_indices_[i];
    }
  }
  dictionary_data_.clear();
  dictionary_indices_.clear();
  location_weightings_.clear();

  // Making room in the vectors for the dictionary/index data.
  dictionary_data_.resize(dictionaries.size());
  dictionary_indices_.resize(dictionaries.size());
  location_weightings_.resize(dictionaries.size());

  // Creating the approximate nearest neighbor indices for codeword
  // matching. This (optionally) uses multiple threads in the case
  // that multiple dictionaries were supplied.
  std::vector<boost::thread*> thread_pool;
  for (size_t dictionary_id = 0; dictionary_id < dictionaries.size();
       ++dictionary_id) {
    if (dictionaries[dictionary_id].centroid_size() == 0) {
      return false;
    }
    sjm::util::PollForAvailablePoolSpace(
        num_threads_, 10, &thread_pool);
    boost::thread* t = new boost::thread(
        &SpatialPyramidBuilder::InitADictionary,
        this,
        boost::ref(dictionaries),
        dictionary_id);
    thread_pool.push_back(t);
  }
  sjm::util::JoinWithPool(&thread_pool);
  return true;
}

// This is the worker function for the dictionary building. It is run
// multi-threaded by the Init function.
void SpatialPyramidBuilder::InitADictionary(
    const std::vector<sjm::codebooks::Dictionary>& dictionaries,
    const size_t dictionary_id) {
  int dimensions = dictionaries[dictionary_id].centroid(0).bin_size();
  location_weightings_[dictionary_id] =
      dictionaries[dictionary_id].location_weighting();

  // Store the new dictionary data in a matrix.
  flann::Matrix<float>* data = new flann::Matrix<float>(
      new float[dictionaries[dictionary_id].centroid_size() *
                dimensions],
      dictionaries[dictionary_id].centroid_size(), dimensions);
  for (int i = 0; i < dictionaries[dictionary_id].centroid_size(); ++i) {
    for (int j = 0; j < dictionaries[dictionary_id].centroid(i).bin_size();
         ++j) {
      (*data)[i][j] = dictionaries[dictionary_id].centroid(i).bin(j);
    }
  }
  const float kBuildWeight = 0;
  const float kMemoryWeight = 0;
  const float kSampleFraction = 0.5;
  const float kAccuracy = 0.95;
  const flann::AutotunedIndexParams params(
      kAccuracy, kBuildWeight, kMemoryWeight, kSampleFraction);
  // Create and build a new index with the new data.
  flann::Index<flann::L2<float> >* index =
      new flann::Index<flann::L2<float> >(*data, params);
  index->buildIndex();

  dictionary_data_[dictionary_id] = data;
  dictionary_indices_[dictionary_id] = index;
}

void SpatialPyramidBuilder::BuildPyramid(
    const sjm::sift::DescriptorSet& descriptors,
    const int num_levels,
    int k,
    const PoolingStrategy pooling_strategy,
    sjm::spatial_pyramid::SpatialPyramid* pyramid) const {
  CHECK_GT(dictionary_data_.size(), 0);
  pyramid->Clear();

  // Determine the final histogram dimensions (the sum of all
  // dictionary sizes).
  int total_histogram_dimensions = 0;
  for (size_t dictionary_id = 0; dictionary_id < dictionary_data_.size();
       ++dictionary_id) {
    total_histogram_dimensions += dictionary_data_[dictionary_id]->rows;
  }

  // Construct an empty pyramid with the proper geometry.
  int grid_size = 1;
  for (int i = 0; i < num_levels; ++i) {
    sjm::spatial_pyramid::PyramidLevel* level = pyramid->add_level();
    level->set_rows(grid_size);
    level->set_columns(grid_size);
    for (int row = 0; row < grid_size; ++row) {
      for (int col = 0; col < grid_size; ++col) {
        sjm::spatial_pyramid::SparseVectorFloat* histogram =
            level->add_histogram();
        histogram->set_non_sparse_length(total_histogram_dimensions);
      }
    }
    grid_size *= 2;
  }

  // If there are no descriptors, just return the empty pyramid.
  if (descriptors.sift_descriptor_size() == 0) {
    return;
  }

  // For each dictionary, we'll add elements to each bin. Normalizing
  // the histograms from each dictionary separately (in the case of
  // average pooling) before concatenating them together in each
  // spatial bin.
  int histogram_index_offset = 0;
  for (size_t dictionary_id = 0; dictionary_id < dictionary_indices_.size();
       ++dictionary_id) {
    // Cap k at the size of the dictionary.
    int capped_k =
        std::min(k, static_cast<int>(dictionary_data_[dictionary_id]->rows));

    // The feature dimensionality includes at least all of the
    // appearance bins.
    int dimensions = descriptors.sift_descriptor(0).bin_size();
    // If location weighting is used, then the feature also includes
    // two bins for the spatial location.
    if (location_weightings_[dictionary_id] > 0) {
      dimensions += 2;
    }

    // This creates the FLANN query matrix out of the query
    // descriptors. This is done for each dictionary because the
    // location weighting could change between the different
    // dictionaries.
    flann::Matrix<float> query(
        new float[descriptors.sift_descriptor_size() *
                  dimensions],
        descriptors.sift_descriptor_size(),
        dimensions);
    for (int i = 0; i < descriptors.sift_descriptor_size(); ++i) {
      for (int j = 0; j < descriptors.sift_descriptor(i).bin_size(); ++j) {
        query[i][j] = descriptors.sift_descriptor(i).bin(j);
      }
      if (location_weightings_[dictionary_id] > 0) {
        query[i][dimensions - 2] =
            descriptors.sift_descriptor(i).x() * 127 *
            location_weightings_[dictionary_id];
        query[i][dimensions - 1] =
            descriptors.sift_descriptor(i).y() * 127 *
            location_weightings_[dictionary_id];
      }
    }

    // These objects will hold the nearest neighbor lookup results.
    flann::Matrix<int> indices(
        new int[query.rows * capped_k], query.rows, capped_k);
    flann::Matrix<float> dists(
        new float[query.rows * capped_k], query.rows, capped_k);

    dictionary_indices_[dictionary_id]->knnSearch(
        query, indices, dists, capped_k,
        flann::SearchParams(flann::FLANN_CHECKS_AUTOTUNED));

    int grid_size = 1;
    float grid_width = 1.0;
    for (int level_id = 0; level_id < num_levels; ++level_id) {
      sjm::spatial_pyramid::PyramidLevel* level =
          pyramid->mutable_level(level_id);
      for (int row = 0; row < grid_size; ++row) {
        for (int col = 0; col < grid_size; ++col) {
          sjm::spatial_pyramid::SparseVectorFloat* histogram =
              level->mutable_histogram(row * grid_size + col);
          // Accumulate the entries for this histogram.
          map<int, float> sparse_histogram;
          for (int d = 0; d < descriptors.sift_descriptor_size(); ++d) {
            float x = descriptors.sift_descriptor(d).x();
            float y = descriptors.sift_descriptor(d).y();
            // Make sure the descriptor is in the bin we're making a
            // histogram for.
            if (x >= col * grid_width && x < (col + 1) * grid_width &&
                y >= row * grid_width && y < (row + 1) * grid_width) {
              // This, combined with the normalization in the match
              // kernel that we use, is the average-pooling operation.
              float soft_assignment_normalizer = 0;
              std::vector<float> accumulations;
              for (int i = 0; i < capped_k; ++i) {
                float dist_squared = dists[d][i] / 16129.0;
                // Get a Gaussian weighting to this descriptor.
                float weight = std::exp(-beta_ * dist_squared);
                // Store the unnormalized weight that the histogram bin
                // at index indices[i] will get accumulated by.
                accumulations.push_back(weight);
                soft_assignment_normalizer += weight;
              }
              // Normalize the weights. This normalization is just
              // across the local nearest neighbors for determining the
              // updates to the histogram caused by this
              // descriptor. Normalization of the histogram happens
              // later.
              for (int i = 0; i < capped_k; ++i) {
                if (soft_assignment_normalizer != 0) {
                  accumulations[i] /= soft_assignment_normalizer;
                }
              }
              // Accumulate the histogram bins with the normalized weights.
              for (int i = 0; i < capped_k; ++i) {
                if (pooling_strategy == AVERAGE_POOLING) {
                  // If average pooling, keep a sum, it will be
                  // normalized later.
                  sparse_histogram[indices[d][i]] += accumulations[i];
                } else if (pooling_strategy == MAX_POOLING) {
                  // If max pooling, just keep track of the max.
                  sparse_histogram[indices[d][i]] =
                      std::max(sparse_histogram[indices[d][i]],
                               accumulations[i]);
                }
              }
            }
          }
          // If average pooling, normalize the histogram bins now. (If
          // were max pooling, nothing needs to be done.)
          if (pooling_strategy == AVERAGE_POOLING) {
            // TODO(sanchom): Look at using std::accumulate here.
            float histogram_sum = 0;
            for (map<int, float>::const_iterator it =
                     sparse_histogram.begin();
                 it != sparse_histogram.end(); ++it) {
              histogram_sum += it->second;
            }
            for (map<int, float>::iterator it = sparse_histogram.begin();
                 it != sparse_histogram.end(); ++it) {
              it->second /= histogram_sum;
            }
          }
          // Move them into the protcol buffer.
          for (map<int, float>::const_iterator it = sparse_histogram.begin();
               it != sparse_histogram.end(); ++it) {
            sjm::spatial_pyramid::SparseValueFloat* sparse_value =
                histogram->add_value();
            sparse_value->set_index(histogram_index_offset + it->first);
            sparse_value->set_value(it->second);
          }
        }
      }
      grid_size *= 2;
      grid_width /= 2;
    }

    histogram_index_offset += dictionary_data_[dictionary_id]->rows;

    delete[] query.ptr();
    delete[] indices.ptr();
    delete[] dists.ptr();
  }
}

void SpatialPyramidBuilder::BuildSingleLevel(
    const sjm::sift::DescriptorSet& descriptors,
    const int level,
    int k,
    const PoolingStrategy pooling_strategy,
    sjm::spatial_pyramid::SpatialPyramid* pyramid) const {
  CHECK_EQ(dictionary_data_.size(), 1) <<
      "Single level pyramids other than level 0 are not implemented "
      "for multiple dictionaries.";
  // Cap k at the size of the dictionary.
  if (k > static_cast<int>(dictionary_data_[0]->rows)) {
    k = dictionary_data_[0]->rows;
  }
  pyramid->Clear();
  // If there are no descriptors in the query, construct an empty
  // pyramid with the proper geometry.
  if (descriptors.sift_descriptor_size() == 0) {
    int grid_size = 1 << level;
    sjm::spatial_pyramid::PyramidLevel* pyramid_level = pyramid->add_level();
    pyramid_level->set_rows(grid_size);
    pyramid_level->set_columns(grid_size);
    for (int row = 0; row < grid_size; ++row) {
      for (int col = 0; col < grid_size; ++col) {
        pyramid_level->add_histogram();
      }
    }
    return;
  }

  int dimensions = descriptors.sift_descriptor(0).bin_size();
  if (location_weightings_[0] > 0) {
    dimensions += 2;
  }

  flann::Matrix<float> query(
      new float[descriptors.sift_descriptor_size() *
                dimensions],
      descriptors.sift_descriptor_size(),
      dimensions);
  for (int i = 0; i < descriptors.sift_descriptor_size(); ++i) {
    for (int j = 0; j < descriptors.sift_descriptor(i).bin_size(); ++j) {
      query[i][j] = descriptors.sift_descriptor(i).bin(j);
    }
    if (location_weightings_[0] > 0) {
      query[i][dimensions - 2] =
          descriptors.sift_descriptor(i).x() * 127 * location_weightings_[0];
      query[i][dimensions - 1] =
          descriptors.sift_descriptor(i).y() * 127 * location_weightings_[0];
    }
  }
  flann::Matrix<int> indices(new int[query.rows * k], query.rows, k);
  flann::Matrix<float> dists(new float[query.rows * k], query.rows, k);
  dictionary_indices_[0]->knnSearch(
      query, indices, dists, k, flann::SearchParams(1));

  int grid_size = 1 << level;
  float grid_width = 1.0f / grid_size;
  // TODO(sanchom): Refactor this common section of code out.
  sjm::spatial_pyramid::PyramidLevel* pyramid_level = pyramid->add_level();
  pyramid_level->set_rows(grid_size);
  pyramid_level->set_columns(grid_size);
  for (int row = 0; row < grid_size; ++row) {
    for (int col = 0; col < grid_size; ++col) {
      sjm::spatial_pyramid::SparseVectorFloat* histogram =
          pyramid_level->add_histogram();
      // Accumulate the entries for this histogram.
      map<int, float> sparse_histogram;
      for (int d = 0; d < descriptors.sift_descriptor_size(); ++d) {
        float x = descriptors.sift_descriptor(d).x();
        float y = descriptors.sift_descriptor(d).y();
        // Make sure the descriptor is in the bin we're making a
        // histogram for.
        if (x >= col * grid_width && x < (col + 1) * grid_width &&
            y >= row * grid_width && y < (row + 1) * grid_width) {
          // This, combined with the normalization in the match
          // kernel that we use, is the average-pooling operation.
          float soft_assignment_normalizer = 0;
          std::vector<float> accumulations;
          for (int i = 0; i < k; ++i) {
            float dist_squared = dists[d][i] / 16129.0;
            // Get a Gaussian weighting to this descriptor.
            float weight = std::exp(-beta_ * dist_squared);
            // Store the unnormalized weight that the histogram bin
            // at index indices[i] will get accumulated by.
            accumulations.push_back(weight);
            soft_assignment_normalizer += weight;
          }
          // Normalize the weights. This normalization is just
          // across the local nearest neighbors for determining the
          // updates to the histogram caused by this
          // descriptor. Normalization of the histogram happens
          // later.
          for (int i = 0; i < k; ++i) {
            if (soft_assignment_normalizer != 0) {
              accumulations[i] /= soft_assignment_normalizer;
            }
          }
          // Accumulate the histogram bins with the normalized weights.
          for (int i = 0; i < k; ++i) {
            if (pooling_strategy == AVERAGE_POOLING) {
              // If average pooling, keep a sum, it will be normalized later.
              sparse_histogram[indices[d][i]] += accumulations[i];
            } else if (pooling_strategy == MAX_POOLING) {
              // If max pooling, just keep track of the max.
              sparse_histogram[indices[d][i]] =
                  std::max(sparse_histogram[indices[d][i]], accumulations[i]);
            }
          }
        }
      }
      // If average pooling, normalize the histogram bins now. (If
      // were max pooling, nothing needs to be done.)
      if (pooling_strategy == AVERAGE_POOLING) {
        // TODO(sanchom): Look at using std::accumulate here.
        float histogram_sum = 0;
        for (map<int, float>::const_iterator it =
                 sparse_histogram.begin();
             it != sparse_histogram.end(); ++it) {
          histogram_sum += it->second;
        }
        for (map<int, float>::iterator it = sparse_histogram.begin();
             it != sparse_histogram.end(); ++it) {
          it->second /= histogram_sum;
        }
      }
      // Move them into the protcol buffer.
      for (map<int, float>::const_iterator it = sparse_histogram.begin();
           it != sparse_histogram.end(); ++it) {
        sjm::spatial_pyramid::SparseValueFloat* sparse_value =
            histogram->add_value();
        sparse_value->set_index(it->first);
        sparse_value->set_value(it->second);
      }
    }
  }

  delete[] query.ptr();
  delete[] indices.ptr();
  delete[] dists.ptr();
}
}}  // namespace.
