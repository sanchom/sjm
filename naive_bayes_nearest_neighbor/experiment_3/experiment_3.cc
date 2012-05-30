// Copyright (c) 2011, Sancho McCann

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

#include <cstdlib>
#include <ctime>
#include <map>
#include <string>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "flann/flann.hpp"

#include "naive_bayes_nearest_neighbor/merged_classifier.h"
#include "sift/sift_descriptors.pb.h"
#include "sift/sift_util.h"
#include "util/util.h"

DEFINE_double(alpha, 0,
              "The location weighting.");
DEFINE_string(results_file, "results.txt",
              "The destination of our results.");
DEFINE_int32(num_train, 15,
             "The number of training images per class.");
DEFINE_int32(num_test, 15,
             "The number of test images per class.");
DEFINE_int32(trees, 4,
             "The number of trees to use for the FLANN search.");
DEFINE_int32(checks, 1,
             "The number of checks to use in the FLANN search.");
DEFINE_string(features_directory, "/var/tmp/sanchom/caltech_local",
              "The directory where the pre-extracted features are.");
DEFINE_int32(k, 5,
             "The number of local nearest neighbors to use.");
DEFINE_int32(b, 1,
             "The (k+b)th neighbor is used to estimate background distance.");
DEFINE_string(category_list, "",
              "A list of category names, found as sub-directories in "
              "the features directory.");
DEFINE_double(subsample, 1.0,
              "A subsample fraction to use for query descriptors.");
DEFINE_bool(full_results, false,
            "Output results from every class, not just the total average.");

using std::map;
using std::string;
using std::vector;

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  std::srand(std::time(NULL));

  CHECK(!FLAGS_category_list.empty()) << "--category_list is required.";
  CHECK_GE(FLAGS_k, 2) << "--k needs to be at least 2.";
  vector<string> categories;
  sjm::util::ReadLinesFromFileIntoVectorOrDie(FLAGS_category_list,
                                              &categories);

  // Get list of files from each category, by looking in the
  // features_directory, constructing the classifier and testing lists
  // as we go.
  sjm::nbnn::MergedClassifier classifier;
  classifier.SetClassifierParams(
      FLAGS_k, FLAGS_k + FLAGS_b, FLAGS_alpha, FLAGS_checks, FLAGS_trees);
  map<string, vector<string> > testing_files;
  boost::filesystem::path root(FLAGS_features_directory);
  for (size_t i = 0; i < categories.size(); ++i) {
    LOG(INFO) << "Loading data for category " << categories[i] << ".";
    boost::filesystem::path stem(categories[i]);
    boost::filesystem::path category_directory = root / stem;

    vector<string> file_list;
    for (boost::filesystem::directory_iterator it(category_directory);
         it != boost::filesystem::directory_iterator();
         ++it) {
      file_list.push_back(it->path().string());
    }
    int num_test =
        std::min(static_cast<int>(file_list.size() - FLAGS_num_train),
                 FLAGS_num_test);
    std::random_shuffle(file_list.begin(), file_list.end());

    // Images 0 to FLAGS_num_train-1 go into the training set.
    for (int j = 0; j < FLAGS_num_train; ++j) {
      sjm::sift::DescriptorSet d;
      sjm::sift::ReadDescriptorSetFromFile(file_list[j], &d);
      classifier.AddData(categories[i], d);
    }

    // Images FLAGS_num_train to FLAGS_num_train + num_test - 1 go in
    // the test set.
    vector<string> test_list;
    for (int j = FLAGS_num_train; j < FLAGS_num_train + num_test; ++j) {
      test_list.push_back(file_list[j]);
    }
    testing_files[categories[i]] = test_list;
  }
  classifier.BuildIndex();

  float mean_accuracy = 0;
  float classes_processed = 0;
  for (map<string, vector<string> >::const_iterator it = testing_files.begin();
       it != testing_files.end(); ++it) {
    string true_category = it->first;
    vector<string> test_list = it->second;
    float correct_this_category = 0;
    float total_this_category = 0;
    BOOST_FOREACH(string test_file, test_list) {
      LOG(INFO) << "Testing " << test_file << ".";
      sjm::sift::DescriptorSet descriptors;
      sjm::sift::ReadDescriptorSetFromFile(test_file, &descriptors);
      sjm::nbnn::Result result = classifier.Classify(descriptors,
                                                     FLAGS_subsample);
      if (result.category == true_category) {
        correct_this_category += 1;
      }
      total_this_category += 1;
      // This logs a running update of the mean accuracy across all
      // classes seen so far, including the current estimated accuracy
      // for the current class.
      LOG(INFO) << "Predicted " << result.category <<
          ". Cumulative mean accuracy = " <<
          ((correct_this_category / total_this_category) + mean_accuracy) /
          (classes_processed + 1) << ".";
    }
    float class_accuracy = correct_this_category / total_this_category;
    // Write this class's accuracy safely to the output file.
    const int kBufferSize = 256;
    char buffer[kBufferSize];
    std::snprintf(buffer, kBufferSize, "%s %f\n",
                  true_category.c_str(), class_accuracy);
    sjm::util::AppendStringToFileOrDie(FLAGS_results_file, buffer);
    mean_accuracy += class_accuracy;
    classes_processed += 1;
  }
  mean_accuracy /= testing_files.size();
  // Write the mean accuracy safely to the output file.
  FILE* f = fopen(sjm::util::expand_user(FLAGS_results_file).c_str(), "a");
  CHECK(f != NULL) << "Error opening " << FLAGS_results_file << ".";
  fprintf(f, "%s %f\n", "total", mean_accuracy);
  fclose(f);

  return 0;
}
