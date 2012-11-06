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

// For usage, see ./spatial_pyramid_builder --helpshort

#include <string>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"
#include "boost/thread.hpp"

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "codebooks/dictionary.pb.h"
#include "sift/sift_descriptors.pb.h"
#include "sift/sift_util.h"
#include "spatial_pyramid/spatial_pyramid.pb.h"
#include "spatial_pyramid/spatial_pyramid_builder.h"
#include "util/util.h"

DEFINE_string(codebooks, "",
              "The paths to the codebooks to use for quantization, separated by"
              " commas (no spaces)");
DEFINE_string(
    input, "",
    "One of 'directory:<dirname>', 'list:<textfile>', or 'file:<siftfile>'.");
DEFINE_int32(levels, 1,
             "The number of spatial pyramid levels to produce.");
DEFINE_int32(single_level, -1,
             "If set to a non-negative value, only this single level of the "
             "spatial pyramid is produced. "
             "This is incompatible with levels > 1");
DEFINE_int32(k, 1,
             "The locality of the soft assignment. To get hard assignment, set "
             "k == 1 (the default).");
DEFINE_string(pooling, "AVERAGE_POOLING",
              "Either AVERAGE_POOLING or MAX_POOLING. This defines the way "
              "features are pooled within each histogram bin.");
DEFINE_int32(thread_limit, 1,
             "The number of threads to use for multithreaded sections.");
DEFINE_bool(clobber, false,
            "Overwrite existing pyramids.");

using std::string;
using std::vector;
using sjm::spatial_pyramid::SpatialPyramid;
using sjm::spatial_pyramid::SpatialPyramidBuilder;

// This is the worker function for the multi-threaded pyramid
// building, when converting many files at a time.
void DoConversion(
    const SpatialPyramidBuilder& builder,
    const sjm::sift::DescriptorSet d,
    const string destination,
    const int levels,
    const int soft_assignment_locality,
    const sjm::spatial_pyramid::PoolingStrategy pooling_strategy) {
  SpatialPyramid pyramid;
  if (FLAGS_single_level < 0) {
    builder.BuildPyramid(
        d, levels, soft_assignment_locality, pooling_strategy, &pyramid);
  } else {
    builder.BuildSingleLevel(
        d, FLAGS_single_level, soft_assignment_locality, pooling_strategy,
        &pyramid);
  }
  // Write pyramid to appropriate output location.
  string serialized_pyramid;
  pyramid.SerializeToString(&serialized_pyramid);
  LOG(INFO) << "Writing " << destination << ".";
  sjm::util::WriteStringToFileOrDie(destination, serialized_pyramid);
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  CHECK(!FLAGS_codebooks.empty()) << "--codebooks is required.";
  CHECK(FLAGS_single_level < 0 || FLAGS_levels == 1) <<
      "You've requested multiple levels, AND specified a single level.";

  vector<string> codebook_paths;
  boost::split(codebook_paths, FLAGS_codebooks, boost::is_any_of(","));
  vector<sjm::codebooks::Dictionary> codebooks;
  for (size_t i = 0; i < codebook_paths.size(); ++i) {
    sjm::codebooks::Dictionary codebook;
    string codebook_string;
    sjm::util::ReadFileToStringOrDie(codebook_paths[i], &codebook_string);
    CHECK(codebook.ParseFromString(codebook_string));
    codebooks.push_back(codebook);
  }

  SpatialPyramidBuilder builder;
  builder.Init(codebooks, FLAGS_thread_limit);

  sjm::spatial_pyramid::PoolingStrategy pooling_strategy =
      sjm::spatial_pyramid::AVERAGE_POOLING;
  if (FLAGS_pooling == "AVERAGE_POOLING") {
    pooling_strategy = sjm::spatial_pyramid::AVERAGE_POOLING;
  } else if (FLAGS_pooling == "MAX_POOLING") {
    pooling_strategy = sjm::spatial_pyramid::MAX_POOLING;
  } else {
    LOG(ERROR) << FLAGS_pooling << " is not an implemented pooling strategy.";
  }

  // Parse the input file structure.
  vector<string> input_parts;
  boost::split(input_parts, FLAGS_input, boost::is_any_of(":"));
  if (input_parts[0] == "directory") {
    // We will recursively walk through the directory tree for files
    // matching *.sift, converting each to a spatial_pyramid.
    LOG(ERROR) <<
        "'directory:<dirname>' specification for input is not implemented.";
  } else if (input_parts[0] == "list") {
    // We will read lines from the text file, converting the files
    // specified on each line to spatial_pyramids.
    string file_data;
    sjm::util::ReadFileToStringOrDie(sjm::util::expand_user(input_parts[1]),
                                     &file_data);
    vector<string> unexpanded_input_files;
    boost::split(unexpanded_input_files, file_data, boost::is_any_of("\n"));
    vector<string> file_list;
    BOOST_FOREACH(string t, unexpanded_input_files) {
      if (!t.empty()) {
        file_list.push_back(sjm::util::expand_user(t));
      }
    }
    vector<boost::thread*> thread_list;
    for (size_t i = 0; i < file_list.size(); ++i) {
      sjm::sift::DescriptorSet d;
      sjm::sift::ReadDescriptorSetFromFile(file_list[i], &d);
      string dest =
          boost::filesystem::path(file_list[i])
          .replace_extension(".pyramid").string();
      if (boost::filesystem::exists(dest) && !FLAGS_clobber) {
        LOG(INFO) << dest <<
            " already exists. Use --clobber option to overwrite.";
        continue;
      }
      sjm::util::PollForAvailablePoolSpace(FLAGS_thread_limit, 1, &thread_list);
      boost::thread* t = new boost::thread(
          DoConversion, boost::ref(builder), d, dest,
          FLAGS_levels, FLAGS_k, pooling_strategy);
      thread_list.push_back(t);
    }
    // Join with any remaining threads.
    sjm::util::JoinWithPool(&thread_list);
  } else if (input_parts[0] == "file") {
    // We will read input from the single file convert it to a spatial
    // pyramid.
    sjm::sift::DescriptorSet d;
    sjm::sift::ReadDescriptorSetFromFile(
        sjm::util::expand_user(input_parts[1]), &d);
    string dest =
        boost::filesystem::path(sjm::util::expand_user(input_parts[1]))
        .replace_extension(".pyramid").string();
    DoConversion(builder, d, dest, FLAGS_levels, FLAGS_k, pooling_strategy);
  }

  return 0;
}
