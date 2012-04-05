// Copyright 2011 Sancho McCann
// Author: Sancho McCann

#ifndef NAIVE_BAYES_NEAREST_NEIGHBOR_MERGED_CLASSIFIER_H_
#define NAIVE_BAYES_NEAREST_NEIGHBOR_MERGED_CLASSIFIER_H_

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "glog/logging.h"

// TODO(sanchom): Extract Result to a common header.
#include "naive_bayes_nearest_neighbor/nbnn_classifier.h"
#include "sift/sift_descriptors.pb.h"

namespace sjm {
namespace nbnn {

class MergedClassifier {
 public:
  MergedClassifier() :
      nearest_neighbors_(1),
      background_index_(2),
      alpha_(0), checks_(1), data_size_(0),
      index_built_(false), data_(NULL), data_dimensions_(0), index_(NULL),
      params_set_(false) {}

  ~MergedClassifier() {
    if (data_) {
      delete[] data_->ptr();
      delete data_;
    }
    if (index_) {
      delete index_;
    }
  }

  void SetClassifierParams(const int nearest_neighbors,
                           const int background_index,
                           const float alpha,
                           const int checks,
                           const int trees) {
    nearest_neighbors_ = nearest_neighbors;
    background_index_ = background_index;
    alpha_ = alpha;
    checks_ = checks;
    trees_ = trees;
    params_set_ = true;
  }

  void AddData(const std::string& class_name,
               const sjm::sift::DescriptorSet& descriptors) {
    CHECK(params_set_) << "Must SetClassifierParams() before adding data.";
    class_set_.insert(class_name);
    if (descriptors.sift_descriptor_size() > 0 && data_dimensions_ == 0) {
      data_dimensions_ = descriptors.sift_descriptor(0).bin_size();
      if (alpha_ > 0) {
        data_dimensions_ += 2;
      }
    } else if (descriptors.sift_descriptor_size() == 0) {
      // No data to add. Do nothing.
      return;
    }

    if (!data_) {
      // On the initial AddData, we just allocate enough storage for
      // 100% of the descriptors.
      data_ = new flann::Matrix<uint8_t>(
          new uint8_t[descriptors.sift_descriptor_size() * data_dimensions_],
          descriptors.sift_descriptor_size(), data_dimensions_);
    } else {
      // On subsequent additions, we double the size of the storage
      // until there's enough storage for the 100% of the additional
      // data.
      int required_data_size = data_size_ + descriptors.sift_descriptor_size();
      // TODO(sanchom): Handle the case where doubling the memory
      // allocation would fail, and adaptively back-off the requested
      // amount by calling new(nothrow).
      while (data_->rows < required_data_size) {
        uint64_t new_size = data_->rows * 2;
        LOG(INFO) << "Growing data matrix to " << new_size * data_dimensions_ <<
            " bytes";
        flann::Matrix<uint8_t>* larger_data =
            new flann::Matrix<uint8_t>(new uint8_t[new_size * data_dimensions_],
                                     new_size, data_dimensions_);
        // Copy old data into larger data space.
        std::copy((*data_)[0],
                  (*data_)[data_->rows - 1] + (data_->cols),
                  (*larger_data)[0]);
        // Delete smaller data structure.
        delete[] data_->ptr();
        delete data_;
        // Re-assign pointer to point to new, larger data structure.
        data_ = larger_data;
      }
    }

    // Put the descriptors into the data.
    for (int i = 0; i < descriptors.sift_descriptor_size(); ++i) {
      int converted_length =
          sjm::sift::ConvertProtobufDescriptorToWeightedArray(
              descriptors.sift_descriptor(i), alpha_,
              (*data_)[data_size_]);
      CHECK(converted_length == data_dimensions_) <<
          "Adding data with inconsistent dimensions.";
      class_vector_.push_back(class_name);
      ++data_size_;
    }
  }

  int DataSize() const {
    return data_size_;
  }

  void BuildIndex() {
    // First, truncate the data to the actual usage. This unfortunately
    // requires allocating additional memory, just to get rid of memory.
    flann::Matrix<uint8_t>* truncated_matrix =
        new flann::Matrix<uint8_t>(new uint8_t[data_size_ * data_dimensions_],
                                   data_size_, data_dimensions_);
    // Copy the data into smaller data space.
    std::copy((*data_)[0],
              (*data_)[data_size_ - 1] + (data_->cols),
              (*truncated_matrix)[0]);
    // Delete original, larger data structure.
    delete[] data_->ptr();
    delete data_;
    // Re-assign pointer to new, smaller data structure.
    data_ = truncated_matrix;
    // TODO(sanchom): Make num trees a parameter.
    index_ = new flann::Index<flann::L2<uint8_t> >(
        *data_, flann::KDTreeIndexParams(trees_));
    index_->buildIndex();
    index_built_ = true;
  }

  Result Classify(const sjm::sift::DescriptorSet& descriptor_set,
                  const float subsample_percentage) const {
    CHECK(index_built_) << "Must call .BuildIndex() before .Classify()";
    // We'll fetch background_index_ features for estimation, capped
    // at the data_size_.
    int b =
        std::min(data_size_, static_cast<uint64_t>(background_index_));
    // We'll use nearest_neighbors_ for foreground, capped at b - 1.
    int k =
        std::min(b - 1, nearest_neighbors_);
    // Set up the class distance accumulator.
    std::map<std::string, float> category_totals;
    for (std::set<std::string>::const_iterator it = class_set_.begin();
         it != class_set_.end();
         ++it) {
      category_totals[*it] = 0;
    }
    // Set up the data for the batch query.
    // First, create a temp array for up to as many descriptors as 100%.
    const size_t kTempSize =
        descriptor_set.sift_descriptor_size() * data_dimensions_;
    uint8_t* temp =
        new uint8_t[kTempSize];
    // Put a subsample of the data into the temp array.
    int next_matrix_index = 0;
    for (int i = 0; i < descriptor_set.sift_descriptor_size(); ++i) {
      if (std::rand() / static_cast<float>(RAND_MAX) < subsample_percentage) {
        sjm::sift::ConvertProtobufDescriptorToWeightedArray(
            descriptor_set.sift_descriptor(i),
            alpha_,
            temp + (next_matrix_index * data_dimensions_));
        ++next_matrix_index;
      }
    }
    // Move the actually used data from the temp array into one that
    // fits.  We don't need to delete this later because it's cleaned up
    // when we delete[] batch_query->data.
    const size_t kQuerySize =
        next_matrix_index * data_dimensions_;
    uint8_t* query_array =
        new uint8_t[kQuerySize];
    std::copy(temp, temp + (next_matrix_index * data_dimensions_),
              query_array);
    delete[] temp;

    // Set up the query for k + 1 nn.
    flann::Matrix<uint8_t> batch_query =
        flann::Matrix<uint8_t>(query_array,
                               next_matrix_index,
                               data_dimensions_);
    flann::Matrix<int> nn_index(new int[batch_query.rows * b],
                                batch_query.rows, b);
    flann::Matrix<float> dists(new float[batch_query.rows * b],
                               batch_query.rows, b);
    // Execute the query, getting indices and dists.
    index_->knnSearch(batch_query, nn_index, dists, b,
                      flann::SearchParams(checks_));
    // For each row, make a category adjustment map
    for (size_t row = 0; row < dists.rows; ++row) {
      std::map<std::string, float> category_distances;
      // The k+1st neighbor is used for the background distance.
      float background_distance = dists[row][b - 1] / 16129.0;
      for (size_t neighbor = 0; neighbor < k; ++neighbor) {
        // Find the category of this neighbor.
        int neighbor_index = nn_index[row][neighbor];
        std::string neighbor_class = class_vector_[neighbor_index];
        // If it's not already been seen,
        if (category_distances.find(neighbor_class) ==
            category_distances.end()) {
          // Put its distance in the map, substracting the background
          // distance.

          // This scales down the distance to be as if the original values
          // had been in [0,1] instead of in [0,127]. Useful in order to
          // avoid overflow errors in some of the probability estimate
          // models. (16129 = 127 * 127
          float distance_squared = dists[row][neighbor] / 16129.0;
          category_distances[neighbor_class] =
              distance_squared - background_distance;
        }
      }
      // Now, adjust the distance totals.
      for (std::map<std::string, float>::const_iterator it =
               category_distances.begin();
           it != category_distances.end();
           ++it) {
        category_totals[it->first] += it->second;
      }
    }
    delete[] batch_query.ptr();
    delete[] nn_index.ptr();
    delete[] dists.ptr();

    // Get the result.
    std::string best_class = "";
    float smallest_distance = 99999999999;
    for (std::map<std::string, float>::const_iterator it =
             category_totals.begin();
         it != category_totals.end(); ++it) {
      if (it->second < smallest_distance) {
        best_class = it->first;
        smallest_distance = it->second;
      }
    }
    Result result;
    result.category = best_class;
    return result;
  }
 private:
  int nearest_neighbors_;
  int background_index_;
  float alpha_;
  int checks_;
  uint64_t data_size_;
  bool index_built_;
  flann::Matrix<uint8_t>* data_;
  int data_dimensions_;
  flann::Index<flann::L2<uint8_t> >* index_;
  bool params_set_;
  int trees_;
  std::vector<std::string> class_vector_;
  std::set<std::string> class_set_;
};
}}  // Namespace.

#endif  // NAIVE_BAYES_NEAREST_NEIGHBOR_MERGED_CLASSIFIER_H_
