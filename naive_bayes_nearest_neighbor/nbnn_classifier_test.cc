// Copyright 2011 Sancho McCann
// Authors: Sancho McCann

#include "naive_bayes_nearest_neighbor/nbnn_classifier.h"

#include <vector>

#include "glog/logging.h"
#include "gtest/gtest.h"

#include "sift/sift_descriptors.pb.h"

using std::vector;

class NbnnClassifierTest : public ::testing::Test {
 protected:
  void SetUp() {
    class_1_data_ =
        new flann::Matrix<uint8_t>(new uint8_t[100 * 128], 100, 128);
    class_2_data_ =
        new flann::Matrix<uint8_t>(new uint8_t[100 * 128], 100, 128);
    for (int i = 0; i < 100; ++i) {
      for (int j = 0; j < 128; ++j) {
        (*class_1_data_)[i][j] = 1;
        (*class_2_data_)[i][j] = 5;
      }
    }
    class_1_index_ =
        new flann::Index<flann::L2<uint8_t> >(*class_1_data_,
                                              flann::KDTreeIndexParams(1));
    class_2_index_ =
        new flann::Index<flann::L2<uint8_t> >(*class_2_data_,
                                              flann::KDTreeIndexParams(1));
    class_1_index_->buildIndex();
    class_2_index_->buildIndex();
  }

  void TearDown() {
    // The index is not deleted because it is owned by the
    // classifier. The classifier deletes the indics on destruction.
    delete[] class_1_data_->ptr();
    delete[] class_2_data_->ptr();
    delete class_1_data_;
    delete class_2_data_;
  }

  sjm::sift::DescriptorSet GetFakeDescriptorSet() const {
    sjm::sift::DescriptorSet descriptor_set;
    for (int i = 0; i < 10; ++i) {
      sjm::sift::SiftDescriptor* d = descriptor_set.add_sift_descriptor();
      for (int j = 0; j < 128; ++j) {
        d->add_bin(2);
      }
    }
    return descriptor_set;
  }
 protected:
  flann::Index<flann::L2<uint8_t> >* class_1_index_;
  flann::Index<flann::L2<uint8_t> >* class_2_index_;
  flann::Matrix<uint8_t>* class_1_data_;
  flann::Matrix<uint8_t>* class_2_data_;
};

TEST_F(NbnnClassifierTest,
       TestConstruction) {
  sjm::nbnn::NbnnClassifier<flann::Index<flann::L2<uint8_t> > > classifier;
  ASSERT_EQ(0, classifier.GetNumClasses());
  ASSERT_TRUE(classifier.GetClassList().empty());
}

TEST_F(NbnnClassifierTest,
       TestAddClasses) {
  sjm::nbnn::NbnnClassifier<flann::Index<flann::L2<uint8_t> > > classifier;
  classifier.AddClass("Class 1", class_1_index_);
  ASSERT_EQ(1, classifier.GetNumClasses());
  ASSERT_EQ(1, classifier.GetClassList().size());
  ASSERT_EQ("Class 1", classifier.GetClassList()[0]);
  classifier.AddClass("Class 2", class_2_index_);
  ASSERT_EQ(2, classifier.GetNumClasses());
  ASSERT_EQ(2, classifier.GetClassList().size());
  ASSERT_EQ("Class 2", classifier.GetClassList()[1]);
}

TEST_F(NbnnClassifierTest,
       TestClassify) {
  sjm::nbnn::NbnnClassifier<flann::Index<flann::L2<uint8_t> > > classifier;
  int nearest_neighbors = 1;
  float alpha = 0;
  int checks = 32;
  classifier.SetClassificationParams(nearest_neighbors, alpha, checks);
  classifier.AddClass("Class 1", class_1_index_);
  classifier.AddClass("Class 2", class_2_index_);
  sjm::sift::DescriptorSet descriptor_set = GetFakeDescriptorSet();
  ASSERT_EQ("Class 1", classifier.Classify(descriptor_set).category);
}

// TODO(sanchom): Write a test for the subsampled classify call.

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
