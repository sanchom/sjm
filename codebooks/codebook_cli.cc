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

// This is a command line tool to interface between saved .sift files
// and the codebook builder.

// Usage: ./codebook_cli --help

#include <string>

#include "boost/algorithm/string.hpp"
#include "boost/foreach.hpp"

#include "flann/flann.hpp"

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "codebooks/codebook_builder.h"
#include "codebooks/dictionary.pb.h"
#include "sift/sift_descriptors.pb.h"
#include "sift/sift_util.h"
#include "util/util.h"

DEFINE_string(
    input, "",
    "One of 'directory:<dirname>', 'list:<textfile>', or 'file:<siftfile>.");
DEFINE_string(
    output, "",
    "The name of the output path for the dictionary.");
DEFINE_int32(max_descriptors, 0,
             "The maximum number of descriptors to load for clustering. If "
             "0 or negative, all the descriptors available are loaded. If "
             "positive, the data is subsampled so approximately that many "
             "descriptors are loaded.");
DEFINE_int32(clusters, 0, "Number of clusters.");
DEFINE_double(location_weighting, 0,
              "The weighting to give the the spatial x, and y dimensions "
              "during clustering.");
DEFINE_double(accuracy, 1.0, "Accuracy of cluster assignment during k-means.");
DEFINE_int32(iterations, 11, "Number of k-means iterations.");
DEFINE_string(
    initialization, "KMEANSPP",
    "KMEANSPP, SUBSAMPLED_KMEANSPP, or RANDOM");
DEFINE_string(
    stats_file, "",
    "A file to which stats will be written.");

using std::string;
using std::vector;

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  CHECK(!FLAGS_input.empty()) << "--input is a required argument.";
  CHECK(!FLAGS_output.empty()) << "--output is a required argument.";
  CHECK_GE(FLAGS_clusters, 1) <<
      "--clusters must be specified and greater than 0.";

  sjm::codebooks::CodebookBuilder builder;

  sjm::codebooks::KMeansInitialization initialization;
  if (FLAGS_initialization == "KMEANSPP") {
    initialization = sjm::codebooks::KMEANS_PP;
  } else if (FLAGS_initialization == "SUBSAMPLED_KMEANSPP") {
    initialization = sjm::codebooks::SUBSAMPLED_KMEANS_PP;
  } else if (FLAGS_initialization == "RANDOM") {
    initialization = sjm::codebooks::KMEANS_RANDOM;
  } else {
    LOG(FATAL) << "Unhandled initialization option.";
  }

  // Parse the input file structure.
  vector<string> input_parts;
  boost::split(input_parts, FLAGS_input, boost::is_any_of(":"));
  if (input_parts[0] == "directory") {
    // We will recursively walk through the directory tree for files
    // matching *.sift, adding each to the codebook builder.
    LOG(ERROR) <<
        "'directory:<dirname>' specification for input is not implemented.";
  } else if (input_parts[0] == "list") {
    // We will read lines from the text file, loading the specified
    // files, and adding each to the codebook builder.
    vector<string> file_list;
    string file_data;
    sjm::util::ReadFileToStringOrDie(input_parts[1], &file_data);
    vector<string> unexpanded_input_files;
    boost::split(unexpanded_input_files, file_data, boost::is_any_of("\n"));
    BOOST_FOREACH(string t, unexpanded_input_files) {
      if (!t.empty()) {
        file_list.push_back(sjm::util::expand_user(t));
      }
    }
    float percentage_to_load = 1.0;
    if (FLAGS_max_descriptors > 0) {
      // Count number of descriptors in training set.
      int total_descriptors = 0;
      for (size_t i = 0; i < file_list.size(); ++i) {
        sjm::sift::DescriptorSet d;
        sjm::sift::ReadDescriptorSetFromFile(file_list[i], &d);
        total_descriptors += d.sift_descriptor_size();
      }
      percentage_to_load =
          static_cast<float>(FLAGS_max_descriptors) /
          static_cast<float>(total_descriptors);
      percentage_to_load = std::min(1.0f, percentage_to_load);
    }

    for (size_t i = 0; i < file_list.size(); ++i) {
      sjm::sift::DescriptorSet d;
      sjm::sift::ReadDescriptorSetFromFile(file_list[i], &d);
      LOG(INFO) << "Adding data from " << file_list[i] << " (" <<
          d.sift_descriptor_size() << ").";
      // TODO(sanchom): Move location_weighting option to
      // builder.Init() since this shouldn't change with each file.
      builder.AddData(d, percentage_to_load, FLAGS_location_weighting);
    }
    LOG(INFO) << "Clustering " << builder.DataSize() << " descriptors.";
  } else if (input_parts[0] == "file") {
    // We will read input from the single file and add it to the
    // codebook builder.
    sjm::sift::DescriptorSet descriptors;
    sjm::sift::ReadDescriptorSetFromFile(sjm::util::expand_user(input_parts[1]),
                                         &descriptors);
    // TODO(sanchom): Implement max_descriptors for single files.
    builder.AddData(descriptors, 1.0, FLAGS_location_weighting);
  }

  double kmeans_metric = 0;
  vector<int> cluster_sizes;
  if (!FLAGS_stats_file.empty()) {
    builder.ClusterApproximately(
        FLAGS_clusters, FLAGS_iterations, FLAGS_accuracy,
        initialization,
        &kmeans_metric,
        &cluster_sizes);
    LOG(INFO) << "k-means metric: " << kmeans_metric;
    for (size_t i = 0; i < cluster_sizes.size(); ++i) {
      LOG(INFO) << "Cluster size: " << cluster_sizes[i];
    }
  } else {
    builder.ClusterApproximately(
        FLAGS_clusters, FLAGS_iterations, FLAGS_accuracy,
        initialization, NULL, NULL);
  }
  sjm::codebooks::Dictionary dictionary;
  builder.GetDictionary(&dictionary);
  if (!FLAGS_stats_file.empty()) {
    // Allocate enough space for every count, plus a comma and a
    // space.
    const size_t kBufferSize = 1024 * cluster_sizes.size() * 7;
    char buffer[kBufferSize];
    int offset = snprintf(buffer, kBufferSize, "%f, ", kmeans_metric);
    for (size_t i = 0; i < cluster_sizes.size() - 1; ++i) {
      offset += snprintf(buffer + offset, kBufferSize, "%d, ",
                         cluster_sizes[i]);
    }
    snprintf(buffer + offset, kBufferSize, "%d\n",
             cluster_sizes[cluster_sizes.size() - 1]);
    sjm::util::AppendStringToFileOrDie(FLAGS_stats_file,
                                       buffer);
  }
  // TODO(sanchom): Move the responsibility for setting this protobuf
  // field to the dictionary builder.
  dictionary.set_location_weighting(FLAGS_location_weighting);
  string serialized_dictionary;
  CHECK(dictionary.SerializeToString(&serialized_dictionary));
  sjm::util::WriteStringToFileOrDie(FLAGS_output, serialized_dictionary);
  return 0;
}
