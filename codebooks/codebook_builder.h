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

// This class takes descriptors and builds a codebook.

#ifndef CODEBOOKS_CODEBOOK_BUILDER_H_
#define CODEBOOKS_CODEBOOK_BUILDER_H_

#include <vector>

#include "gflags/gflags.h"

#include "flann/flann.hpp"

DECLARE_string(initialization_checkpoint_file);

// Forward-declarations.
namespace sjm {
namespace sift {
class DescriptorSet;
}}

namespace sjm {
namespace codebooks {
class Dictionary;
}}

namespace sjm {
namespace codebooks {

enum KMeansInitialization {
  KMEANS_RANDOM,  // Kmeans is initialized using random points from the data.
  KMEANS_PP,  // Kmeans++ initialization is used (Google it).
  SUBSAMPLED_KMEANS_PP  // Kmeans++ is used on a random 10% of the data.
};

class CodebookBuilder {
 public:
  CodebookBuilder() : data_dimensions_(0), matrix_usage_(0) {
    data_ = NULL;
    centroids_ = NULL;
  }
  ~CodebookBuilder() {
    if (data_) {
      delete[] data_->ptr();
      delete data_;
    }
    if (centroids_) {
      delete[] centroids_->ptr();
      delete centroids_;
    }
  }
  // Adds data from the descriptor set to the object's database for
  // use in a later call to Cluster(). The percentage is a hint and
  // only approximate.
  //
  // If location_weighting > 0, two extra dimensions are added to the
  // descriptor bins that represent the x and y spatial locations of
  // the descriptor.
  void AddData(const sjm::sift::DescriptorSet& descriptors,
               const float percentage,
               const float location_weighting);
  // Clusters the data into num_clusters centroids with num_iterations
  // of k-means.
  void Cluster(const int num_clusters, const int num_iterations);
  // Like ::Cluster(), but uses an approximate nearest neighbor search
  // for the cluster assignment at each iteration.
  //
  // Optional metric and sizes arguments can be used to get a metric
  // related to cluster quality and a vector of cluster cardinalities.
  void ClusterApproximately(const int num_clusters, const int num_iterations,
                            const float accuracy,
                            const KMeansInitialization initialization,
                            double* metric = NULL,
                            std::vector<int>* sizes = NULL);
  // Populates the Dictionary object with the centroids from
  // clustering.
  //
  // If location_weighting > 0 in AddData(), then the dictionary will
  // have two extra dimensions representing the x and y locations,
  // scaled to be in [0,127] * location_weighting.
  void GetDictionary(Dictionary* dictionary) const;
  // Returns the number of data points that have been added.
  int DataSize() const;
 private:
  int data_dimensions_;
  int matrix_usage_;
  flann::Matrix<float>* data_;
  flann::Matrix<float>* centroids_;
};
}}  // namespace.

#endif  // CODEBOOKS_CODEBOOK_BUILDER_H_
