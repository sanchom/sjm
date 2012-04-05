// Copyright 2011 Sancho McCann
// Authors: Sancho McCann

#ifndef NAIVE_BAYES_NEAREST_NEIGHBOR_NBNN_CLASSIFIER_H_
#define NAIVE_BAYES_NEAREST_NEIGHBOR_NBNN_CLASSIFIER_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "flann/flann.hpp"
#include "sift/sift_descriptors.pb.h"

namespace sjm {
namespace nbnn {

struct Result {
  std::string category;
};

// The IndexType can be any class that provides the following function:
//
// IndexType::knnSearch(flann::Matrix<uint8_t>, flann::Matrix<int>,
//                      flann::Matrix<float>, int, flann::SearchParams)
template <class IndexType>
class NbnnClassifier {
 public:
  NbnnClassifier() : nearest_neighbors_(1), alpha_(0), checks_(1) {}
  ~NbnnClassifier();
  int GetNumClasses() const;
  const std::vector<std::string>& GetClassList() const;
  void SetClassificationParams(const int nearest_neighbors,
                               const float alpha,
                               const int checks);
  // The index object is owned by this NbnnClassifier object now. It
  // will be deleted properly upon destruction.
  void AddClass(const std::string& class_name,
                IndexType* index);
  Result Classify(const sjm::sift::DescriptorSet& descriptor_set) const;
  Result Classify(const sjm::sift::DescriptorSet& descriptor_set,
                  const float subsample_percentage) const;
 private:
  std::vector<std::string> class_list_;
  std::map<std::string, IndexType*> indices_;
  int nearest_neighbors_;
  float alpha_;
  float checks_;
};
}}  // Namespace.

#include "naive_bayes_nearest_neighbor/nbnn_classifier-inl.h"

#endif  // NAIVE_BAYES_NEAREST_NEIGHBOR_NBNN_CLASSIFIER_H_
