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

#ifndef NAIVE_BAYES_NEAREST_NEIGHBOR_NBNN_CLASSIFIER_INL_H_
#define NAIVE_BAYES_NEAREST_NEIGHBOR_NBNN_CLASSIFIER_INL_H_

#include <algorithm>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "glog/logging.h"

#include "flann/flann.hpp"

#include "sift/sift_descriptors.pb.h"
#include "sift/sift_util.h"

namespace sjm {
namespace nbnn {

template <typename IndexType>
NbnnClassifier<IndexType>::~NbnnClassifier() {
  for (typename std::map<std::string, IndexType*>::iterator it =
           indices_.begin();
       it != indices_.end();
       ++it) {
    delete it->second;
  }
  indices_.clear();
}

template <typename IndexType>
int NbnnClassifier<IndexType>::GetNumClasses() const {
  return static_cast<int>(class_list_.size());
}

template <typename IndexType>
const
std::vector<std::string>& NbnnClassifier<IndexType>::GetClassList() const {
  return class_list_;
}

template <typename IndexType>
void NbnnClassifier<IndexType>::SetClassificationParams(
    const int nearest_neighbors,
    const float alpha,
    const int checks) {
  nearest_neighbors_ = nearest_neighbors;
  alpha_ = alpha;
  checks_ = checks;
}

template <typename IndexType>
void NbnnClassifier<IndexType>::AddClass(const std::string& class_name,
                                         IndexType* index) {
  class_list_.push_back(class_name);
  CHECK(indices_.find(class_name) == indices_.end()) <<
      "Attempting to insert a class twice.";
  indices_[class_name] = index;
}

template <typename IndexType>
Result NbnnClassifier<IndexType>::Classify(
    const sjm::sift::DescriptorSet& descriptor_set) const {
  return Classify(descriptor_set, 1.0);
}

template <typename IndexType>
Result NbnnClassifier<IndexType>::Classify(
    const sjm::sift::DescriptorSet& descriptor_set,
    const float subsample_percentage) const {
  // Set up accumulator.
  std::map<std::string, float> distance_totals;
  for (size_t i = 0; i < class_list_.size(); ++i) {
    distance_totals[class_list_[i]] = 0;
  }
  // Get the dimensions.
  uint8_t *destination = new uint8_t[130];
  int dimensions =
      sjm::sift::ConvertProtobufDescriptorToWeightedArray(
          descriptor_set.sift_descriptor(0), alpha_, destination);
  delete[] destination;
  // Set up the data for the batch query.
  // First, create a temp array for up to 100% of the descriptors.
  uint8_t* temp =
      new uint8_t[descriptor_set.sift_descriptor_size() * dimensions];
  // Put a subsample of the data into the temp array.
  int next_matrix_index = 0;
  for (int i = 0; i < descriptor_set.sift_descriptor_size(); ++i) {
    if (std::rand() / static_cast<float>(RAND_MAX) < subsample_percentage) {
      sjm::sift::ConvertProtobufDescriptorToWeightedArray(
          descriptor_set.sift_descriptor(i),
          alpha_,
          temp + (next_matrix_index * dimensions));
      ++next_matrix_index;
    }
  }

  // Move the actually used data from the temp array into one that
  // fits.  We don't need to delete this later because it's cleaned up
  // when we delete[] batch_query->data.
  uint8_t* query_array =
      new uint8_t[next_matrix_index * dimensions];
  std::copy(temp, temp + (next_matrix_index * dimensions),
            query_array);
  delete[] temp;

  flann::Matrix<uint8_t> batch_query =
      flann::Matrix<uint8_t>(query_array,
                             next_matrix_index,
                             dimensions);

  // NN query Result matrices.
  flann::Matrix<int> nn_index(new int[batch_query.rows * nearest_neighbors_],
                              batch_query.rows, nearest_neighbors_);
  flann::Matrix<float> dists(new float[batch_query.rows * nearest_neighbors_],
                             batch_query.rows, nearest_neighbors_);
  // Shuffle the order in which the classes are queried. This does not
  // affect the results, but is important for avoiding resource
  // contention when this is run in parallel and IndexType is a
  // connection to a FLANN server.
  std::vector<std::string> query_ordering = class_list_;
  std::random_shuffle(query_ordering.begin(), query_ordering.end());
  // This is the implmentation of the NBNN algorithm.
  for (size_t i = 0; i < query_ordering.size(); ++i) {
    IndexType* class_index = indices_.find(query_ordering[i])->second;
    // For all query descriptors, find their nearest neighbor(s) in
    // this class_index.
    class_index->knnSearch(
        batch_query, nn_index, dists, nearest_neighbors_,
        flann::SearchParams(checks_));
    // Total up the squared distances from each query descriptor to
    // their nearest neighbors in this class.
    for (size_t j = 0; j < dists.rows; ++j) {
      // This scaling is necessary because descriptor values are
      // stored in [0,127] (for space savings), so we divide the
      // distance squared (dists[j][0]) by 127^2. This avoids
      // overflows if these distances are later used in probability
      // estimate models.
      float distance_squared = dists[j][0] / 16129.0;
      distance_totals[query_ordering[i]] += distance_squared;
    }
  }
  delete[] batch_query.ptr();
  delete[] nn_index.ptr();
  delete[] dists.ptr();
  std::string best_class = "";
  float smallest_distance = 99999999999;
  for (std::map<std::string, float>::const_iterator it =
           distance_totals.begin();
       it != distance_totals.end(); ++it) {
    if (it->second < smallest_distance) {
      best_class = it->first;
      smallest_distance = it->second;
    }
  }
  Result r;
  r.category = best_class;
  return r;
}
}}  // Namespace.

#endif  // NAIVE_BAYES_NEAREST_NEIGHBOR_NBNN_CLASSIFIER_INL_H_
