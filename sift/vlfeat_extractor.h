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

// This file provides a class that wraps calls to
// Oxford's vlfeat implementation of dense sift extraction

#pragma once

#include "extractor.h"

namespace sjm {
  namespace sift {
    class DescriptorSet; // Forward declaration
    class ExtractionParameters; // Forward declaration

    // Wrapper class for vlfeat dense sift extraction
    class VlFeatExtractor : public Extractor {
    public:
      // Initializes the image and the parameters for the extraction
      // See sift_descriptors.proto for descriptions of the parameters
      VlFeatExtractor(const cv::Mat & image, ExtractionParameters parameters);
      ~VlFeatExtractor();
      // Re-sets the parameters for the extraction
      void set_parameters(ExtractionParameters parameters);
      // Performs the extraction on the image with the options specified by
      // the parameters.
      // Returns a sjm::sift::DescriptorSet defined in sift_descriptors.proto
      DescriptorSet Extract() const;
   private:
      // This is the minimum width in pixels of a SIFT bin. A SIFT descriptor
      // describes an area covered by a 4x4 bin arrangement. So,
      // bin_size = radius / 2.  Restricting the minimum_raduis = 8
      // means the minimum bin size = 4.  It's possible for the user
      // to specify a larger minimum radius through
      // extraction_parameters_.minimum_radius().
      static const int minimum_bin_size_ = 4;
      // magnif is a constant from the original sift implementation
      // used to convert the bin size to a scale value. See the
      // discussion at http://www.vlfeat.org/overview/dsift.html:
      //      DSIFT specifies the descriptor size by a single
      //      parameter, size, which controls the size of a SIFT
      //      spatial bin in pixels. In the standard SIFT descriptor,
      //      the bin size is related to the SIFT keypoint scale by a
      //      multiplier, denoted magnif below, which defaults to
      //      3. As a consequence, a DSIFT descriptor with bin size
      //      equal to 5 corresponds to a SIFT keypoint of scale
      //      5/3=1.66.
      static const float magnif_;
    };
  } // namespace sift
} // namespace sjm
