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

// This class implements the building of Spatial Pyramids given a set
// of sift features, and a codebook (or codebooks).

// The original work implemented in this class is SPATIALLY LOCAL
// CODING. If you supply a dictionary that uses weighted location
// information, features are coded as in Spatially Local Coding. You
// should use just the bag-of-words level of the Spatial Pyramid in
// this case.

// It also implements the basic SPM by Lazebnik et al. It implements
// Local Soft Assignment Coding by Liu et al. It allows for either
// average pooling or max pooling in each bin. It allows for 1, 2, 3,
// or more pyramid levels. It allows building of just a single one of
// those levels (for example you could build just the second level of
// the standard SPM).

#ifndef SPATIAL_PYRAMID_SPATIAL_PYRAMID_BUILDER_H_
#define SPATIAL_PYRAMID_SPATIAL_PYRAMID_BUILDER_H_

#include <vector>

#include "flann/flann.hpp"

// Forward declarations.
namespace sjm {
namespace codebooks {
class Dictionary;
}}

namespace sjm {
namespace sift {
class DescriptorSet;
}}

namespace sjm {
namespace spatial_pyramid {

// Forward declaration.
class SpatialPyramid;

enum PoolingStrategy {
  AVERAGE_POOLING = 0,
  MAX_POOLING = 1
};

class SpatialPyramidBuilder {
 public:
  SpatialPyramidBuilder() {
    // TODO(sanchom): Change this to a parameter. This is the weight
    // decay in the local soft assignment coding.
    beta_ = 10;
  }
  ~SpatialPyramidBuilder() {
    // TODO(sanchom): Refactor this out to a private FreeData
    // function.
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
  }
  // Prepares the object for building spatial pyramids using the
  // provided dictionary (dictionaries). If more than one dictionary
  // is provided, features are coded using all dictionaries
  // simultaneously, and the histograms are concatenated within each
  // spatial bin. Use dictionaries built with location weighting > 0
  // to use Spatially Local Coding.
  //
  // num_threads gives the maximum number of threads that will be used
  // when initializing the indices and when performing the searches
  // across multiple dictionaries.
  bool Init(const std::vector<sjm::codebooks::Dictionary>& dictionaries,
            const int num_threads);
  // Turns descriptor sets into spatial pyramids using the previously
  // provided dictionary. The pyramid will have num_levels levels,
  // with the first level being the bag-of-words level (1x1), the
  // second level being split into a 2x2 grid, the third level being
  // split into a 4x4 grid, etc.
  //
  // If dictionaries had location weighting, then choose num_levels=1,
  // k=10, pooling_strategy=MAX_POOLING to build the Spatially Local
  // Coding representation.
  //
  // Params
  // - descriptors: The descriptors to quantize into the spatial pyramid.
  // - num_levels: The number of hierarchical levels in the pyramid.
  // - k: The locality of the nearest neighbor search and soft assignment.
  //      If k == 1, this is just hard codeword assignment.
  // - pooling_strategy: Choose either average pooling or max pooling.
  // - pyramid: A pointer to a pyramid into which the result will be stored.
  void BuildPyramid(const sjm::sift::DescriptorSet& descriptors,
                    const int num_levels,
                    int k,
                    const PoolingStrategy pooling_strategy,
                    sjm::spatial_pyramid::SpatialPyramid* pyramid) const;
  // Builds a single level of a spatial pyramid using the previously
  // provided dictionary. The pyramid will have a single level,
  // specified by the 'level' parameter. If 'level' = 0, you'll get
  // the bag-of-words level (1x1). If 'level' = 1, you'll get the 2x2
  // level. Etc. Other arguments are as in BuildPyramid.
  void BuildSingleLevel(const sjm::sift::DescriptorSet& descriptors,
                        const int level,
                        int k,
                        const PoolingStrategy pooling_strategy,
                        sjm::spatial_pyramid::SpatialPyramid* pyramid) const;

 private:
  void InitADictionary(
      const std::vector<sjm::codebooks::Dictionary>& dictionaries,
      const size_t dictionary_id);
  std::vector<flann::Matrix<float>* > dictionary_data_;
  std::vector<flann::Index<flann::L2<float> >* > dictionary_indices_;
  std::vector<float> location_weightings_;
  float beta_;
  int num_threads_;
};

}}  // namespace.

#endif  // SPATIAL_PYRAMID_SPATIAL_PYRAMID_BUILDER_H_
