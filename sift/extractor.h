// Copyright 2010 Sancho McCann
// Author: Sancho McCann

// This file contains the definition for the abstract base class for
// sift extraction strategies. Client code should hold pointers to the
// sjm::sift::Extractor type, with polymorphic behaviour determined by
// which subclass actually was instantiated.

#pragma once

#include <opencv2/opencv.hpp>

#include <iostream>

#include "sift_descriptors.pb.h"

namespace sjm {
  namespace sift {
    // The abstract base class for sift extractors.
    class Extractor {
    public:
      // Virtual destructor so subclasses can override this behaviour
      virtual ~Extractor() { }

      // TODO (sanchom): check type of CvMat, enforce CV_8UC1
      // Initializes a pointer to an image to process
      // Responsibility for management of this memory stays with the client
      void set_image(const cv::Mat & image) {
	image_ = image.clone();
	image_initialized_ = true;
      }
      
      // Default implementation for setting parameters, with no checking
      // Parameters defined in sift_descriptors.proto
      virtual void set_parameters(ExtractionParameters parameters) {
	extraction_parameters_ = parameters;
	parameters_initialized_ = true;
      }

      // Returns true if image and parameters are both set properly
      bool IsInitialized() const {
	return image_initialized_ && parameters_initialized_;
      }

      // This must be re-implemented in subclasses to execute the particular
      // extraction strategy
      virtual DescriptorSet Extract() const = 0;
    protected:
      cv::Mat image_; // An open cv image structure
      ExtractionParameters extraction_parameters_; // Defined at sift_descriptors.proto
      bool parameters_initialized_;
    private:
      bool image_initialized_;    
    };
  } // namespace sift
} // namespace sjm
