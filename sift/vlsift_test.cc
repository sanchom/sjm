// Copyright (c) 2010, Sancho McCann

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

// This file tests my interpretation of the vlsift dense sift code
// and my wrapper class for that code.

#include <iostream>
#include <sstream>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "gtest/gtest.h"

extern "C" {
#include "vl/dsift.h"
#include "vl/generic.h"
}

#include "sift_descriptors.pb.h"
#include "vlfeat_extractor.h"

class VlSiftLearningTest : public ::testing::Test {
 protected:
  void SetUp() {
    CvMat * test_image = cvLoadImageM("../test_images/seminar.pgm", CV_LOAD_IMAGE_GRAYSCALE);

    unsigned char * original_data = test_image->data.ptr;
    rows = test_image->rows;
    cols = test_image->cols;
    int step = test_image->step / sizeof(unsigned);
    data = new float[rows * cols];
    for (unsigned j = 0; j < rows; ++j) {
      for (unsigned i = 0; i < cols; ++i) {
	data[j * cols + i] = (original_data + i * step)[j] / 255.0;
      }
    }

    cvReleaseMat(&test_image);
  }

  void TearDown() {
    delete[] data;
  }

  float * data;
  unsigned rows;
  unsigned cols;
};

class VlSiftWrapperTest : public ::testing::Test {
protected:
  void SetUp() {
    test_image = cvLoadImageM("../test_images/seminar.pgm", CV_LOAD_IMAGE_GRAYSCALE);
    original_cerr_buffer = std::cerr.rdbuf();
    std::cerr.rdbuf(replacement_cerr_buffer.rdbuf());
  }

  void TearDown() {
    std::cerr.rdbuf(original_cerr_buffer);
    cvReleaseMat(&test_image);
  }

  int CountZeroDescriptors(const sjm::sift::DescriptorSet & descriptor_set) {
    int num_zero_descriptors = 0;
    for (int i = 0; i < descriptor_set.sift_descriptor_size(); ++i) {
      sjm::sift::SiftDescriptor d =
	descriptor_set.sift_descriptor(i);
      bool is_zero_descriptor = true;
      for (int j = 0; j < d.bin_size(); ++j) {
	if (d.bin(j) != 0) {
	  is_zero_descriptor = false;
	  break;
	}
      }
      if (is_zero_descriptor) {
	num_zero_descriptors += 1;
      }
    }
    return num_zero_descriptors;
  }
  
  std::stringstream replacement_cerr_buffer;
  std::streambuf * original_cerr_buffer;
  CvMat * test_image;
};

typedef VlSiftWrapperTest VlSiftWrapperDeathTest;

TEST_F(VlSiftLearningTest, BasicExtractionWorks) {
  VlDsiftFilter * filter = vl_dsift_new_basic(cols, rows, 8, 4);
  ASSERT_GT(vl_dsift_get_keypoint_num(filter), 0);
  vl_dsift_process(filter, data);
  const VlDsiftKeypoint * keypoints = vl_dsift_get_keypoints(filter);
  // For all i, 0 <= keypoint[i].x < cols, 0 <= keypoint[i].y < rows,
  // 0 <= keypoint[i].norm <= 1
  for (int i = 0; i < vl_dsift_get_keypoint_num(filter); ++i) {
    ASSERT_GE(keypoints[i].x, 0) << "keypoint[" << i << "]";
    ASSERT_GE(keypoints[i].y, 0) << "keypoint[" << i << "]";
    ASSERT_LT(keypoints[i].x, cols) << "keypoint[" << i << "]";
    ASSERT_LT(keypoints[i].y, rows) << "keypoint[" << i << "]";
    ASSERT_GE(keypoints[i].norm, 0) << "keypoint[" << i << "]";
    ASSERT_LE(keypoints[i].norm, 1) << "keypoint[" << i << "]";
    
  }
}

TEST_F(VlSiftLearningTest, CoarserGridReturnsFewerDescriptors) {
  // Initially, the grid is 8 x 8
  VlDsiftFilter * filter = vl_dsift_new_basic(cols, rows, 8, 4);
  int coarse_returns = vl_dsift_get_keypoint_num(filter);
  // Now, the grid is 4 x 4
  vl_dsift_set_steps(filter, 4, 4);
  int fine_returns = vl_dsift_get_keypoint_num(filter);
  ASSERT_GT(fine_returns, coarse_returns);
}

TEST_F(VlSiftLearningTest, FlatFilterIsFaster) {
  VlDsiftFilter * filter = vl_dsift_new_basic(cols, rows, 8, 4);

  vl_dsift_set_flat_window(filter, true);
  vl_tic();
  vl_dsift_process(filter, data);
  double flat_time = vl_toc();

  vl_dsift_set_flat_window(filter, false);
  vl_tic();
  vl_dsift_process(filter, data);
  double gaussian_time = vl_toc();

  ASSERT_GT(gaussian_time, flat_time);
}

TEST_F(VlSiftLearningTest, CheckDefaultSettings) {
  int binSize = 6;
  VlDsiftFilter * filter = vl_dsift_new_basic(cols, rows, 8, binSize);
  
  const VlDsiftDescriptorGeometry * geometry = vl_dsift_get_geometry(filter);
  ASSERT_EQ(8, geometry->numBinT);
  ASSERT_EQ(4, geometry->numBinX);
  ASSERT_EQ(4, geometry->numBinY);
  ASSERT_EQ(binSize, geometry->binSizeX);
  ASSERT_EQ(binSize, geometry->binSizeY);  
  ASSERT_FALSE(vl_dsift_get_flat_window(filter));

  int minX;
  int minY;
  int maxX;
  int maxY;
  vl_dsift_get_bounds(filter, &minX, &minY, &maxX, &maxY);
  ASSERT_EQ(0, minX);
  ASSERT_EQ(0, minY);
  ASSERT_EQ(cols - 1, maxX);
  ASSERT_EQ(rows - 1, maxY);
}

TEST_F(VlSiftLearningTest, CheckInterpretationOfGetSteps) {
  int specifiedStep = 8;
  int stepX;
  int stepY;
  VlDsiftFilter * filter = vl_dsift_new_basic(cols, rows, specifiedStep, 4);
  vl_dsift_get_steps(filter, &stepX, &stepY);

  ASSERT_EQ(specifiedStep, stepX);
  ASSERT_EQ(specifiedStep, stepY);
}

TEST_F(VlSiftWrapperTest, ConstructionWorks) {
  sjm::sift::ExtractionParameters parameters;
  sjm::sift::Extractor * extractor = new sjm::sift::VlFeatExtractor(test_image, parameters);
  ASSERT_TRUE(extractor->IsInitialized());
  delete(extractor);
}

TEST_F(VlSiftWrapperTest, ConstructionWarnsWithWrongImplementation) {
  sjm::sift::ExtractionParameters parameters;
  parameters.set_implementation(sjm::sift::ExtractionParameters::KOEN);
  sjm::sift::Extractor * extractor = new sjm::sift::VlFeatExtractor(test_image, parameters);
  ASSERT_TRUE(replacement_cerr_buffer.str().find("Warning") != std::string::npos);
  delete(extractor);
}

TEST_F(VlSiftWrapperTest, BasicExtractionWorks) {
  sjm::sift::ExtractionParameters parameters;
  sjm::sift::Extractor * extractor = new sjm::sift::VlFeatExtractor(test_image, parameters);
  sjm::sift::DescriptorSet descriptor_set = extractor->Extract();
  ASSERT_GT(descriptor_set.sift_descriptor_size(), 0);
  delete(extractor);
}

TEST_F(VlSiftWrapperTest, ObservesMultiscaleParameter) {
  sjm::sift::ExtractionParameters parameters;
  sjm::sift::Extractor * extractor = new sjm::sift::VlFeatExtractor(test_image, parameters);
  sjm::sift::DescriptorSet descriptor_set_multiscale = extractor->Extract();
  parameters.set_multiscale(false);
  extractor->set_parameters(parameters);
  sjm::sift::DescriptorSet descriptor_set_singlescale = extractor->Extract();
  ASSERT_GT(descriptor_set_multiscale.sift_descriptor_size(),
	    descriptor_set_singlescale.sift_descriptor_size());
  delete(extractor);
}

TEST_F(VlSiftWrapperTest, ObservesMinimumRadiusParameter) {
  sjm::sift::ExtractionParameters parameters;
  sjm::sift::Extractor * extractor = new sjm::sift::VlFeatExtractor(test_image, parameters);
  sjm::sift::DescriptorSet descriptor_set_unset = extractor->Extract();
  parameters.set_minimum_radius(8);
  extractor->set_parameters(parameters);
  sjm::sift::DescriptorSet descriptor_set_default = extractor->Extract();
  parameters.set_minimum_radius(12);
  extractor->set_parameters(parameters);
  sjm::sift::DescriptorSet descriptor_set_fewer = extractor->Extract();
  ASSERT_LT(descriptor_set_fewer.sift_descriptor_size(),
	    descriptor_set_default.sift_descriptor_size());
  ASSERT_EQ(descriptor_set_default.sift_descriptor_size(),
	    descriptor_set_unset.sift_descriptor_size());
  delete(extractor);
}

TEST_F(VlSiftWrapperTest, ObservesFractionalXYParameter) {
  sjm::sift::ExtractionParameters parameters;
  parameters.set_fractional_xy(false);
  sjm::sift::Extractor * extractor = new sjm::sift::VlFeatExtractor(test_image, parameters);
  sjm::sift::DescriptorSet descriptor_set_image_coords = extractor->Extract();
  parameters.set_fractional_xy(true);
  extractor->set_parameters(parameters);
  sjm::sift::DescriptorSet descriptor_set_fractional = extractor->Extract();
  // These should give equal sized returns
  ASSERT_EQ(descriptor_set_image_coords.sift_descriptor_size(),
	    descriptor_set_fractional.sift_descriptor_size());
  // Check that the returns for the image_coords request are not restricted to [0,1]
  bool things_greater_than_two = false;
  for (int i = 0; i < descriptor_set_image_coords.sift_descriptor_size(); ++i) {
    sjm::sift::SiftDescriptor d = descriptor_set_image_coords.sift_descriptor(i);
    ASSERT_GE(d.x(), 0);
    ASSERT_LT(d.x(), test_image->cols);
    ASSERT_GE(d.y(), 0);
    ASSERT_LT(d.y(), test_image->rows);
    if (d.x() > 2 || d.y() > 2) {
      things_greater_than_two = true;
    }
  }
  ASSERT_TRUE(things_greater_than_two);

  // Check that the returns for the fractional_xy are restricted to [0,1]
  for (int i = 0; i < descriptor_set_fractional.sift_descriptor_size(); ++i) {
    sjm::sift::SiftDescriptor d = descriptor_set_fractional.sift_descriptor(i);
    ASSERT_GE(d.x(), 0);
    ASSERT_LE(d.x(), 1);
    ASSERT_GE(d.y(), 0);
    ASSERT_LE(d.y(), 1);
  }
}

TEST_F(VlSiftWrapperTest, ObservesResolutionFactorParameter) {
  sjm::sift::ExtractionParameters parameters;
  sjm::sift::Extractor * extractor = new sjm::sift::VlFeatExtractor(test_image, parameters);
  sjm::sift::DescriptorSet descriptor_set_unset = extractor->Extract();
  parameters.set_grid_method(sjm::sift::ExtractionParameters::FIXED_3X3);
  extractor->set_parameters(parameters);
  sjm::sift::DescriptorSet descriptor_set_largest = extractor->Extract();
  parameters.set_grid_method(sjm::sift::ExtractionParameters::FIXED_8X8);
  extractor->set_parameters(parameters);
  sjm::sift::DescriptorSet descriptor_set_8x8 = extractor->Extract();
  parameters.set_grid_method(sjm::sift::ExtractionParameters::SCALED_BIN_WIDTH);
  extractor->set_parameters(parameters);
  sjm::sift::DescriptorSet descriptor_set_bin_width = extractor->Extract();
  parameters.set_grid_method(sjm::sift::ExtractionParameters::SCALED_3X3);
  extractor->set_parameters(parameters);
  // This set should have more than SCALED_BIN_WIDTH, and less than FIXED_3X3
  sjm::sift::DescriptorSet descriptor_set_between = extractor->Extract();
  parameters.set_grid_method(
      sjm::sift::ExtractionParameters::SCALED_DOUBLE_BIN_WIDTH);
  extractor->set_parameters(parameters);
  sjm::sift::DescriptorSet descriptor_set_quarter = extractor->Extract();
  ASSERT_GT(descriptor_set_largest.sift_descriptor_size(),
	    descriptor_set_bin_width.sift_descriptor_size());
  ASSERT_GT(descriptor_set_bin_width.sift_descriptor_size(),
	    descriptor_set_quarter.sift_descriptor_size());
  ASSERT_EQ(descriptor_set_unset.sift_descriptor_size(),
	    descriptor_set_largest.sift_descriptor_size());
  ASSERT_GT(descriptor_set_between.sift_descriptor_size(),
            descriptor_set_bin_width.sift_descriptor_size());
  ASSERT_LT(descriptor_set_between.sift_descriptor_size(),
            descriptor_set_largest.sift_descriptor_size());
  ASSERT_LT(descriptor_set_8x8.sift_descriptor_size(),
            descriptor_set_largest.sift_descriptor_size());
  delete(extractor);
}

TEST_F(VlSiftWrapperTest, ObservesPercentage) {
  sjm::sift::ExtractionParameters parameters;
  sjm::sift::Extractor * extractor = new sjm::sift::VlFeatExtractor(test_image, parameters);
  sjm::sift::DescriptorSet descriptor_set_full = extractor->Extract();
  parameters.set_percentage(0.5);
  extractor->set_parameters(parameters);
  sjm::sift::DescriptorSet descriptor_set_half = extractor->Extract();
  ASSERT_LT(descriptor_set_half.sift_descriptor_size(),
	    descriptor_set_full.sift_descriptor_size());
  ASSERT_LT(descriptor_set_half.sift_descriptor_size(),
            descriptor_set_full.sift_descriptor_size() * 0.6);
  ASSERT_GT(descriptor_set_half.sift_descriptor_size(),
            descriptor_set_full.sift_descriptor_size() * 0.4);
  delete(extractor);
}

TEST_F(VlSiftWrapperTest, ObservesBoundingBoxWithIntegerLocation) {
  sjm::sift::ExtractionParameters parameters;
  sjm::sift::Extractor * extractor = new sjm::sift::VlFeatExtractor(test_image, parameters);
  sjm::sift::DescriptorSet descriptor_set_unbounded = extractor->Extract();
  int top_left_x = 30;
  int top_left_y = 50;
  int bottom_right_x = 100;
  int bottom_right_y = 90;
  parameters.set_top_left_x(top_left_x);
  parameters.set_top_left_y(top_left_y);
  parameters.set_bottom_right_x(bottom_right_x);
  parameters.set_bottom_right_y(bottom_right_y);
  extractor->set_parameters(parameters);
  sjm::sift::DescriptorSet descriptor_set_bounded = extractor->Extract();
  ASSERT_LT(descriptor_set_bounded.sift_descriptor_size(),
	    descriptor_set_unbounded.sift_descriptor_size());
  for (int i = 0; i < descriptor_set_bounded.sift_descriptor_size(); ++i) {
    sjm::sift::SiftDescriptor d = descriptor_set_bounded.sift_descriptor(i);
    ASSERT_GE(d.x(), 0);
    ASSERT_GE(d.y(), 0);
    ASSERT_LE(d.x(), bottom_right_x - top_left_x + 1);
    ASSERT_LE(d.y(), bottom_right_y - top_left_y + 1);
  }
}

TEST_F(VlSiftWrapperTest, ObservesBoundingBoxWithFractionalLocation) {
  sjm::sift::ExtractionParameters parameters;
  sjm::sift::Extractor * extractor = new sjm::sift::VlFeatExtractor(test_image, parameters);  
  int top_left_x = 30;
  int top_left_y = 50;
  int bottom_right_x = 100;
  int bottom_right_y = 90;
  parameters.set_top_left_x(top_left_x);
  parameters.set_top_left_y(top_left_y);
  parameters.set_bottom_right_x(bottom_right_x);
  parameters.set_bottom_right_y(bottom_right_y);
  parameters.set_fractional_xy(true);
  extractor->set_parameters(parameters);
  sjm::sift::DescriptorSet descriptor_set_bounded_fractional =
    extractor->Extract();
  bool x_below_0_25 = false;
  bool x_above_0_75 = false;
  bool y_below_0_25 = false;
  bool y_above_0_75 = false;

  for (int i = 0; i < descriptor_set_bounded_fractional.sift_descriptor_size(); ++i) {
    sjm::sift::SiftDescriptor d = descriptor_set_bounded_fractional.sift_descriptor(i);
    ASSERT_GE(d.x(), 0);
    ASSERT_GE(d.y(), 0);
    ASSERT_LE(d.x(), 1);
    ASSERT_LE(d.y(), 1);
    if (d.x() > 0.75) {
      x_above_0_75 = true;
    }
    if (d.x() < 0.25) {
      x_below_0_25 = true;
    }
    if (d.y() > 0.75) {
      y_above_0_75 = true;
    }
    if (d.y() < 0.25) {
      y_below_0_25 = true;
    }
  }
  ASSERT_TRUE(x_above_0_75);
  ASSERT_TRUE(x_below_0_25);
  ASSERT_TRUE(y_above_0_75);
  ASSERT_TRUE(y_below_0_25);
}

TEST_F(VlSiftWrapperTest, NotAllZero) {
  sjm::sift::ExtractionParameters parameters;
  sjm::sift::Extractor * extractor = new sjm::sift::VlFeatExtractor(test_image, parameters);
  sjm::sift::DescriptorSet descriptor_set = extractor->Extract();
  bool exists_a_non_zero_descriptor = false;
  for (int i = 0; i < descriptor_set.sift_descriptor_size(); ++i) {
    sjm::sift::SiftDescriptor d = descriptor_set.sift_descriptor(i);
    for (int bin_id = 0; bin_id < d.bin_size(); ++bin_id) {
      if (d.bin(bin_id) > 0) {
        exists_a_non_zero_descriptor = true;
        break;
      }
    }
    if (exists_a_non_zero_descriptor) {
      break;
    }
  }
  ASSERT_TRUE(exists_a_non_zero_descriptor);
}

TEST_F(VlSiftWrapperTest, SmoothingCappedAtMaximum) {
  sjm::sift::ExtractionParameters parameters;
  parameters.set_first_level_smoothing(1.8);
  sjm::sift::Extractor * extractor = new sjm::sift::VlFeatExtractor(test_image, parameters);
  sjm::sift::DescriptorSet descriptor_set_smoothed = extractor->Extract();
  ASSERT_FLOAT_EQ(0.6666666,
                  descriptor_set_smoothed.parameters().first_level_smoothing());
}

TEST_F(VlSiftWrapperTest, SmoothedVersionExtractsTheSameNumber) {
  sjm::sift::ExtractionParameters parameters;
  sjm::sift::Extractor * extractor = new sjm::sift::VlFeatExtractor(test_image, parameters);
  sjm::sift::DescriptorSet descriptor_set_smoothed = extractor->Extract();
  parameters.set_first_level_smoothing(0.5);
  extractor->set_parameters(parameters);
  sjm::sift::DescriptorSet descriptor_set_unsmoothed = extractor->Extract();
  ASSERT_EQ(descriptor_set_smoothed.sift_descriptor_size(),
	    descriptor_set_unsmoothed.sift_descriptor_size());
}

TEST_F(VlSiftWrapperTest, SmoothedVersionLessIfDiscardingLowContrast) {
  sjm::sift::ExtractionParameters parameters;
  parameters.set_first_level_smoothing(0.5);
  parameters.set_discard_unnormalized(true);
  parameters.set_normalization_threshold(1.27);
  sjm::sift::Extractor * extractor = new sjm::sift::VlFeatExtractor(test_image, parameters);
  sjm::sift::DescriptorSet descriptor_set_smoothed = extractor->Extract();
  parameters.set_first_level_smoothing(0.0);
  extractor->set_parameters(parameters);
  sjm::sift::DescriptorSet descriptor_set_unsmoothed = extractor->Extract();
  ASSERT_LT(descriptor_set_smoothed.sift_descriptor_size(),
	    descriptor_set_unsmoothed.sift_descriptor_size());
}

TEST_F(VlSiftWrapperTest, NonSmoothedImageReturnsMoreDescriptors) {
  sjm::sift::ExtractionParameters parameters;
  parameters.set_smoothed(false);
  parameters.set_first_level_smoothing(0.5);
  parameters.set_discard_unnormalized(true);
  parameters.set_normalization_threshold(1.5);
  sjm::sift::Extractor* extractor =
      new sjm::sift::VlFeatExtractor(test_image, parameters);
  sjm::sift::DescriptorSet descriptor_set_unsmoothed = extractor->Extract();
  parameters.set_smoothed(true);
  extractor->set_parameters(parameters);
  sjm::sift::DescriptorSet descriptor_set_smoothed = extractor->Extract();
  ASSERT_LT(descriptor_set_smoothed.sift_descriptor_size(),
            descriptor_set_unsmoothed.sift_descriptor_size());  
}

TEST_F(VlSiftWrapperTest, NormalizationAndDiscardTest) {
  sjm::sift::ExtractionParameters parameters;
  sjm::sift::Extractor * extractor = new sjm::sift::VlFeatExtractor(test_image, parameters);
  sjm::sift::DescriptorSet descriptor_set_no_threshold = extractor->Extract();
  parameters.set_discard_unnormalized(true);
  extractor->set_parameters(parameters);
  sjm::sift::DescriptorSet descriptor_set_discard_none = extractor->Extract();
  ASSERT_EQ(descriptor_set_no_threshold.sift_descriptor_size(),
	    descriptor_set_discard_none.sift_descriptor_size());
  parameters.set_normalization_threshold(0.05);
  extractor->set_parameters(parameters);
  sjm::sift::DescriptorSet descriptor_set_thresholded_discard = extractor->Extract();
  ASSERT_LT(descriptor_set_thresholded_discard.sift_descriptor_size(),
	    descriptor_set_no_threshold.sift_descriptor_size());
  parameters.set_discard_unnormalized(false);
  extractor->set_parameters(parameters);
  sjm::sift::DescriptorSet descriptor_set_thresholded_no_discard = extractor->Extract();
  ASSERT_EQ(descriptor_set_no_threshold.sift_descriptor_size(),
	    descriptor_set_thresholded_no_discard.sift_descriptor_size());
  int zero_descriptors_in_original =
    CountZeroDescriptors(descriptor_set_no_threshold);
  // Can't assert this. If a descriptor is found in a zero-contrast area, it is retained
  // and is zero in every bin. It's length is zero, and can't be normalized.
  // ASSERT_EQ(0, zero_descriptors_in_original);
  int zero_descriptors_in_thresholded =
    CountZeroDescriptors(descriptor_set_thresholded_no_discard);
  int num_discarded = descriptor_set_no_threshold.sift_descriptor_size() -
    descriptor_set_thresholded_discard.sift_descriptor_size();
  ASSERT_EQ(num_discarded, zero_descriptors_in_thresholded);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
