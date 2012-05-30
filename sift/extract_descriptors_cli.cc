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

// This file implements a command line interface for SIFT feature
// extraction.
//
// SIFT files end up being written like this:
//
// <length of serialized parameter protobuf> (4-byte integer)
// <serialized parameter protobuf> (length specified by previous)
// <length of serialized descriptorset protobuf> (4-byte integer)
// <serialized descriptorset protobuf> (length specified by previous)

#include <set>
#include <string>
#include <vector>

#include "boost/bind.hpp"
#include "boost/filesystem.hpp"
#include "boost/numeric/conversion/bounds.hpp"

#include "opencv2/opencv.hpp"

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "util/util.h"
#include "sift/extractor.h"
#include "sift/sift_descriptors.pb.h"
#include "sift/sift_util.h"
#include "sift/vlfeat_extractor.h"

DEFINE_int32(tlx, 0, "The top left x coordinate of the extraction subwindow.");
DEFINE_int32(tly, 0, "The top left y coordinate of the extraction subwindow.");
DEFINE_int32(brx, boost::numeric::bounds<int32_t>::highest(),
             "The bottom right x coordinate of the extraction subwindow.");
DEFINE_int32(bry, boost::numeric::bounds<int32_t>::highest(),
             "The bottom right y coordinate of the extraction subwindow.");
DEFINE_double(normalization_threshold, 0.0,
              "SIFT descriptors with contrast below this are not normalized.");
DEFINE_double(minimum_radius, 0,
              "The minimum SIFT radius to extract. The smallest meaningful "
              "value is 8. Values below 8 give 16x16 SIFT descriptors "
              "(radius of 8).");
DEFINE_double(percentage, 1.0, "Percentage of SIFT descriptors to extract.");
DEFINE_bool(multiscale, true, "Multiscale SIFT extraction.");
DEFINE_string(output_directory, "",
              "An alternate output directory. By default, SIFT is output to "
              "the same directory as the source.");
DEFINE_bool(clobber, false, "Overwrite any existing output files.");
DEFINE_bool(fractional_location, true,
            "Describes the position as values in [0,1] x [0,1] rather than "
            "in absolute pixel locations.");
DEFINE_bool(discard, false,
            "Discard descriptors that fail the constrast threshold test. "
            "See --normalization_threshold for more information.");
DEFINE_double(first_level_smoothing, 0, "This sigma of smoothing will "
              "be applied to the 16x16 level of the scale pyramid.");
DEFINE_bool(smooth, true,
            "If true, smoothing is done as stepping up through the "
            "scale space.");
DEFINE_bool(fast, true, "Use a fast approximation to the original "
            "SIFT descriptor.");
DEFINE_string(grid_type, "FIXED_3X3",
              "One of {FIXED_3X3, FIXED_8X8, SCALED_3X3, SCALED_BIN_WIDTH, "
              "SCALED_DOUBLE_BIN_WIDTH}.");

namespace fs = boost::filesystem;
namespace sift = sjm::sift;
using std::set;
using std::string;
using std::vector;

int main(int argc, char** argv) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  vector<string> input_paths;

  sift::ExtractionParameters sift_parameters;
  sift_parameters.set_normalization_threshold(FLAGS_normalization_threshold);
  sift_parameters.set_discard_unnormalized(FLAGS_discard);
  sift_parameters.set_multiscale(FLAGS_multiscale);
  sift_parameters.set_percentage(FLAGS_percentage);
  sift_parameters.set_minimum_radius(FLAGS_minimum_radius);
  sift_parameters.set_fractional_xy(FLAGS_fractional_location);
  sift_parameters.set_smoothed(FLAGS_smooth);
  sift_parameters.set_fast(FLAGS_fast);

  for (int i = 1; i < argc; ++i) {
    input_paths.push_back(argv[i]);
  }

  if (FLAGS_grid_type == "FIXED_3X3") {
    sift_parameters.set_grid_method(sjm::sift::ExtractionParameters::FIXED_3X3);
  } else if (FLAGS_grid_type == "FIXED_8X8") {
    sift_parameters.set_grid_method(
        sjm::sift::ExtractionParameters::FIXED_8X8);
  } else if (FLAGS_grid_type == "SCALED_3X3") {
    sift_parameters.set_grid_method(
        sjm::sift::ExtractionParameters::SCALED_3X3);
  } else if (FLAGS_grid_type == "SCALED_BIN_WIDTH") {
    sift_parameters.set_grid_method(
        sjm::sift::ExtractionParameters::SCALED_BIN_WIDTH);
  } else if (FLAGS_grid_type == "SCALED_DOUBLE_BIN_WIDTH") {
    sift_parameters.set_grid_method(
        sjm::sift::ExtractionParameters::SCALED_DOUBLE_BIN_WIDTH);
  } else {
    LOG(FATAL) << "--grid_type " << FLAGS_grid_type << " is invalid.";
  }

  sift_parameters.set_normalization_threshold(FLAGS_normalization_threshold);
  sift_parameters.set_top_left_x(FLAGS_tlx);
  sift_parameters.set_top_left_y(FLAGS_tly);
  sift_parameters.set_bottom_right_x(FLAGS_brx);
  sift_parameters.set_bottom_right_y(FLAGS_bry);
  sift_parameters.set_multiscale(FLAGS_multiscale);
  sift_parameters.set_discard_unnormalized(FLAGS_discard);
  sift_parameters.set_percentage(FLAGS_percentage);
  sift_parameters.set_minimum_radius(FLAGS_minimum_radius);
  sift_parameters.set_fractional_xy(FLAGS_fractional_location);
  sift_parameters.set_first_level_smoothing(FLAGS_first_level_smoothing);
  sift_parameters.set_smoothed(FLAGS_smooth);
  sift_parameters.set_fast(FLAGS_fast);

  sift::Extractor * extractor;
  sift_parameters.set_implementation(sjm::sift::ExtractionParameters::VLFEAT);
  extractor = new sift::VlFeatExtractor(cv::Mat(), sift_parameters);

  // Doing the extractions.
  vector<string>::const_iterator it = input_paths.begin();
  for ( ; it != input_paths.end(); ++it ) {
    LOG(INFO) << "Processing " << *it << ".";
    fs::path sift_path = fs::change_extension(*it, ".sift");
    if (FLAGS_output_directory.size() > 0) {
      // Change output path to include user-specified output directory.
      sift_path =
          fs::path(FLAGS_output_directory) / fs::path(sift_path.leaf());
    }

    // Being careful about overwriting already existing data.
    if (!fs::exists(sift_path) || FLAGS_clobber) {
      // The "0" for the second argument forces greyscale loading.
      cv::Mat cv_image = cv::imread(*it, 0);
      if (cv_image.data != NULL) {
        extractor->set_image(cv_image);
        sift::DescriptorSet descriptor_set = extractor->Extract();
        sift::WriteDescriptorSetToFile(descriptor_set, sift_path.string());
        LOG(INFO) << "Wrote " << sift_path.string() << ".";
      } else {
        LOG(ERROR) << "Error loading file.";
      }
    } else {
      LOG(INFO) << sift_path.string() << " already exists.";
    }
  }
  delete(extractor);

  return 0;
}
