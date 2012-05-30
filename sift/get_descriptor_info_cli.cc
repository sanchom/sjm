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

// A command line interface for reporting the details of a .sift file
// as written by sjm::sift::WriteDescriptorSetToFile or
// extract_descriptors_cli.
//
// Usage:
// ./get_descriptor_info_cli [sift_file]
//
// > Rotation invariance: false
// > No-normalize: 0.500000
// > Discard non-normalized descriptors: false
// > Single scale
// > Percentage: 1.000000
// > Min radius: 0.000000
// > Grid resolution equal to 2 x bin width (radius) at any scale.
// > First level smoothing: 0.000000
// > Fractional location
// > Descriptors 768
//
// ./get_descriptor_info_cli --count [sift_file]
//
// > 768

#include <cstdio>
#include <string>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/filesystem/fstream.hpp"
#include "boost/program_options.hpp"

#include "sift/sift_descriptors.pb.h"
#include "sift/sift_util.h"

using std::string;
using std::vector;
namespace sift = sjm::sift;
namespace fs = boost::filesystem;
namespace po = boost::program_options;

enum Mode {
  All,
  Count
};

int main(int argc, char** argv) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  vector<string> inputFiles;
  bool verbose = false;
  Mode mode = All;

  po::options_description command_line_options("Command line options");
  command_line_options.add_options()
      ("help,H", "produce help message")
      ("count,C", "only produce count of descriptors")
      ("verbose,V", "also print out descriptor values")
      ("input,I", po::value<vector<string> >(), "single input image");
  po::positional_options_description positional;
  positional.add("input", -1);

  try {
    po::variables_map vm;
    po::store(
        po::command_line_parser(argc, argv).options(
            command_line_options).positional(positional).run(),
        vm);
    po::notify(vm);

    if (!vm.count("input")) {
      printf("--input argument required.\n");
      exit(1);
    } else {
      inputFiles = vm["input"].as<vector<string> >();
    }
    if (vm.count("verbose")) {
      verbose = true;
    }
    if (vm.count("count")) {
      mode = Count;
    }
  } catch(po::invalid_command_line_syntax) {
    printf("Invalid usage.\n");
    exit(1);
  }

  for (size_t i = 0; i < inputFiles.size(); ++i) {
    const string & inputFile = inputFiles[i];
    if (fs::exists(inputFile)) {
      sift::DescriptorSet descriptor_set;
      sift::ReadDescriptorSetFromFile(inputFile, &descriptor_set);

      if  (mode == All) {
        printf("Rotation invariance: %s\n",
               descriptor_set.parameters().rotation_invariance() ?
               "true" : "false");
        printf("No-normalize: %f\n",
               descriptor_set.parameters().normalization_threshold());
        printf("Discard non-normalized descriptors: %s\n",
               descriptor_set.parameters().discard_unnormalized() ?
               "true" : "false");
        printf("%s\n",
               descriptor_set.parameters().multiscale() ?
               "Multiscale" : "Single scale");
        printf("Percentage: %f\n", descriptor_set.parameters().percentage());
        printf("Min radius: %f\n",
               descriptor_set.parameters().minimum_radius());
        switch (descriptor_set.parameters().grid_method()) {
          case sjm::sift::ExtractionParameters::FIXED_3X3:
            printf("Fixed 3x3 grid (at all scales, if multiscale)\n");
            break;
          case sjm::sift::ExtractionParameters::FIXED_8X8:
            printf("Fixed 8x8 grid (at all scales, if multiscale)\n");
            break;
          case sjm::sift::ExtractionParameters::SCALED_3X3:
            printf("3x3 grid for 16x16 descriptors, "
                   "scaled up for larger descriptors\n");
            break;
          case sjm::sift::ExtractionParameters::SCALED_BIN_WIDTH:
            printf("Grid resolution equal to descriptor bin width (1/2 radius) "
                   "at any scale.\n");
            break;
          case sjm::sift::ExtractionParameters::SCALED_DOUBLE_BIN_WIDTH:
            printf("Grid resolution equal to 2 x bin width (radius) "
                   "at any scale.\n");
            break;
        }
        printf("First level smoothing: %f\n",
               descriptor_set.parameters().first_level_smoothing());
        printf("%s\n",
               descriptor_set.parameters().fractional_xy() ?
               "Fractional location" : "Pixel location");
        printf("Descriptors %d\n", descriptor_set.sift_descriptor_size());
      }

      if ( mode == Count ) {
        printf("%d\n", descriptor_set.sift_descriptor_size());
      }
      if (verbose) {
        for (int d = 0; d < descriptor_set.sift_descriptor_size(); ++d) {
          printf("%s\n",
                 descriptor_set.sift_descriptor(d).DebugString().c_str());
        }
      }
    }
  }
  return 0;
}
