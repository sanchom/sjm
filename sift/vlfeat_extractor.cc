// Copyright 2010 Sancho McCann
// Author: Sancho McCann

// Implementation wrapping my vlfeat dense sift extraction.
// Documentation is in the associated .h file.

#include "sift/vlfeat_extractor.h"

#include <opencv2/opencv.hpp>

#include <tr1/cstdint>

#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "sift/sift_descriptors.pb.h"
extern "C" {
#include "vl/dsift.h"
}

namespace sjm {
  namespace sift {
    const int VlFeatExtractor::minimum_bin_size_;
    VlFeatExtractor::VlFeatExtractor(const cv::Mat & image,
                                     ExtractionParameters parameters) {
      set_image(image);
      set_parameters(parameters);
    }

    VlFeatExtractor::~VlFeatExtractor() {}

    void VlFeatExtractor::set_parameters(ExtractionParameters parameters) {
      if (parameters.has_implementation() &&
          parameters.implementation() != sift::ExtractionParameters::VLFEAT) {
        std::cerr << "Warning: implementation pre-set to something other "<<
            "than VLFEAT, but called VlFeatExtractor()." << std::endl;
      }
      if (parameters.first_level_smoothing() >
          minimum_bin_size_ / magnif_ + 0.0001) {
        std::cerr << "Warning: too much first level smoothing is requested (" <<
            parameters.first_level_smoothing() << "). " <<
            "Smoothing is being capped at " <<
            minimum_bin_size_ / magnif_ << std::endl;
        parameters.set_first_level_smoothing(minimum_bin_size_ / magnif_);
      }
      extraction_parameters_ = parameters;
      extraction_parameters_.set_implementation(
          sift::ExtractionParameters::VLFEAT);
      parameters_initialized_ = true;
    }

    DescriptorSet VlFeatExtractor::Extract() const {
      if (!IsInitialized()) {
        std::cerr << "Extractor not properly initialized." << std::endl;
        exit(1);
      }
      DescriptorSet d;

      // This is the width in pixels of a SIFT bin. A SIFT descriptor
      // describes an area covered by a 4x4 bin arrangement. So,
      // bin_size = radius / 2.  Restricting the minimum_raduis = 8
      // means the minimum bin size = 4.  It's possible for the user
      // to specify a larger minimum radius through
      // extraction_parameters_.minimum_radius().
      int initial_bin_size =
          std::max(minimum_bin_size_,
                   static_cast<int>(extraction_parameters_.minimum_radius() /
                                    2.0f + 0.5f));

      // Create space for the successively smoothed image
      cv::Mat smoothed_image(image_.rows, image_.cols, CV_8UC1);

      // Define the number of scales to extract SIFT at. Multiscale
      // gives 3 scales, otherwise, use a single scale.
      int levels;
      if (extraction_parameters_.multiscale()) {
        levels = 3;
      } else {
        levels = 1;
      }

      // This sets an assumed about of smoothing to exist in the
      // image already such that the first level is smoothed by the
      // amount requested by the user code.
      float assumed_smoothing =
          ((minimum_bin_size_ / magnif_) *
           (minimum_bin_size_ / magnif_)) -
          (extraction_parameters_.first_level_smoothing() *
           extraction_parameters_.first_level_smoothing());

      int bin_size = initial_bin_size;

      // For each scale level, gather descriptors
      for (int level = 0; level < levels; ++level) {
        // Compute the scale associated with this bin size
        float scale = bin_size / magnif_;
        // Compute the sigma needed for the smoothing call.
        // This assumes some initial smoothing simply due to
        // the camera sensor array.
        float sigma =
            std::sqrt(std::max(0.0f, scale * scale - assumed_smoothing));
        // Smooth the original image by by sigma,
        // or just copy it if no smoothing.
        if (extraction_parameters_.smoothed() && sigma > 0) {
          cv::GaussianBlur(image_, smoothed_image,
                           cv::Size(0, 0), sigma);
        } else {
          smoothed_image = image_.clone();
        }

        // Get the data from the smoothed image's 4-byte aligned
        // memory into a contiguous array of memory for vlfeat, also
        // converting the range from [0, 255] to [0,1].
        unsigned char * smoothed_data = smoothed_image.ptr();
        int rows = smoothed_image.rows;
        int cols = smoothed_image.cols;
        // smoothed_image->step gives the full row length in bytes.
        // This may not equal cols * sizeof(unsigned) due to alignment
        // gaps.
        int row_step = smoothed_image.step / sizeof(unsigned char);
        // Move the data from the aligned CvMat representation to the
        // contiguous float array.
        float * smoothed_data_contiguous = new float[rows * cols];
        for (int y = 0; y < rows; ++y) {
          for (int x = 0; x < cols; ++x) {
            smoothed_data_contiguous[y * cols + x] =
              (smoothed_data + y * row_step)[x];
          }
        }
        // Set the step size to 3 pixels, as per Vedaldi and Boiman.
        int step_size = 3;
        float scaling_factor =
            static_cast<float>(bin_size / minimum_bin_size_);
        switch (extraction_parameters_.grid_method()) {
          case sjm::sift::ExtractionParameters::FIXED_3X3:
            step_size = 3;
            break;
          case sjm::sift::ExtractionParameters::FIXED_8X8:
            step_size = 8;
            break;
          case sjm::sift::ExtractionParameters::SCALED_3X3:
            step_size =
                static_cast<int>(scaling_factor * 3 + 0.5);
            break;
          case sjm::sift::ExtractionParameters::SCALED_BIN_WIDTH:
            step_size = bin_size;
            break;
          case sjm::sift::ExtractionParameters::SCALED_DOUBLE_BIN_WIDTH:
            step_size = 2 * bin_size;
            break;
        }

        // Set up the dense sift extractor. Code in this scope is
        // responsible for calling vl_dsift_delete_filter(filter) when
        // done with this filter.
        // Argument order is: (image_width, image_height, steps, bin_size)
        VlDsiftFilter * filter =
          vl_dsift_new_basic(cols, rows, step_size, bin_size);

        // Turns off the Gaussian weighting within SIFT descriptor
        // (negligible difference in accuracy, but much faster).
        vl_dsift_set_flat_window(filter, extraction_parameters_.fast());

        // Use the parameters to determine and set the bounds of the extraction
        int min_x = std::max(0U, extraction_parameters_.top_left_x());
        int min_y = std::max(0U, extraction_parameters_.top_left_y());
        int max_x = std::min(static_cast<unsigned>(cols - 1),
                             extraction_parameters_.bottom_right_x());
        int max_y = std::min(static_cast<unsigned>(rows - 1),
                             extraction_parameters_.bottom_right_y());
        int window_width = max_x - min_x + 1;
        int window_height = max_y - min_y + 1;
        vl_dsift_set_bounds(filter, min_x, min_y, max_x, max_y);

        // Actually do the processing
        vl_dsift_process(filter, smoothed_data_contiguous);
        delete[] smoothed_data_contiguous;

        // Convert the vlfeat representation of the extracted descriptors
        // to our protocol buffer representation.

        int descriptor_size = vl_dsift_get_descriptor_size(filter);
        const VlDsiftKeypoint * keypoints = vl_dsift_get_keypoints(filter);
        const float * descriptors = vl_dsift_get_descriptors(filter);

        for (int descriptor_id = 0;
             descriptor_id < vl_dsift_get_keypoint_num(filter);
             ++descriptor_id) {
          if (rand() / static_cast<float>(RAND_MAX) >=
              extraction_parameters_.percentage()) {
            continue;
          }
          // Make x and y relative to the subwindow top-left
          float x_val = keypoints[descriptor_id].x - min_x;
          float y_val = keypoints[descriptor_id].y - min_y;
          // Optionally make x and y fractional coordinates with
          // (0,0) being the top-left and (1,1) being the bottom right
          if (extraction_parameters_.fractional_xy()) {
            x_val /= static_cast<float>(window_width);
            y_val /= static_cast<float>(window_height);
          }
          if (keypoints[descriptor_id].norm >=
              extraction_parameters_.normalization_threshold()) {
            // If the descriptor passed the normalization threshold,
            // store it as-is
            SiftDescriptor * descriptor = d.add_sift_descriptor();
            descriptor->set_x(x_val);
            descriptor->set_y(y_val);
            descriptor->set_scale(scale);
            for (int bin_id = 0; bin_id < descriptor_size; ++bin_id) {
              // Using the actual values
              // But, multiply them by 127 to move them from [0,1) floats to
              // [0,127] integers
              descriptor->add_bin(static_cast<std::tr1::uint32_t>(
                  (descriptors + descriptor_id * descriptor_size)[bin_id] * 127
                  + 0.5));
            }
          } else if (!extraction_parameters_.discard_unnormalized()) {
            // Otherwise (if the descriptor failed the threshold
            // test), and if we don't just discard the descriptors
            // that failed, we zero them out instead.
            SiftDescriptor * descriptor = d.add_sift_descriptor();
            descriptor->set_x(x_val);
            descriptor->set_y(y_val);
            descriptor->set_scale(scale);
            for (int bin_id = 0; bin_id < descriptor_size; ++bin_id) {
              // Making this a zero-descriptor
              descriptor->add_bin(0);
            }
          }
        }
        vl_dsift_delete(filter);
        // Step up by 1.5 for the next scale (matches Vedaldi's PHOW
        // code).
        bin_size = static_cast<int>(bin_size * 1.5 + 0.5);
      }

      ExtractionParameters * params_to_set = d.mutable_parameters();
      params_to_set->CopyFrom(extraction_parameters_);
      return d;
    }
  }  // namespace sift
}  // namespace sjm
