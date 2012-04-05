// Copyright 2011 Sancho McCann
// Authors: Sancho McCann

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
  // First, create a temp array for up to as many descriptors as 100%.
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

  // Check for each class, the distances to it, and record it in the
  // accumulator.
  flann::Matrix<int> nn_index(new int[batch_query.rows * nearest_neighbors_],
                              batch_query.rows, nearest_neighbors_);
  flann::Matrix<float> dists(new float[batch_query.rows * nearest_neighbors_],
                             batch_query.rows, nearest_neighbors_);
  // Shuffle the order in which the classes are queried.
  std::vector<std::string> query_ordering = class_list_;
  std::random_shuffle(query_ordering.begin(), query_ordering.end());
  for (size_t i = 0; i < query_ordering.size(); ++i) {
    IndexType* index = indices_.find(query_ordering[i])->second;
    index->knnSearch(
        batch_query, nn_index, dists, nearest_neighbors_,
        flann::SearchParams(checks_));
    for (size_t j = 0; j < dists.rows; ++j) {
      // This scales down the distance to be as if the original values
      // had been in [0,1] instead of in [0,127]. Useful in order to
      // avoid overflow errors in some of the probability estimate
      // models. (16129 = 127 * 127)
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
