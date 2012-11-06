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

// This command line tool uses learned SVM models to classify a set of
// test files.

#include <map>
#include <set>
#include <string>
#include <vector>

#include "boost/foreach.hpp"
#include "boost/thread.hpp"

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "spatial_pyramid/spatial_pyramid.pb.h"
#include "spatial_pyramid/spatial_pyramid_kernel.h"
#include "spatial_pyramid/svm/svm.h"
#include "util/util.h"

DEFINE_string(
    training_list, "",
    "A list of the paths to the pyramids to be used as support vectors. "
    "Each line is '<train_file>:<ground_truth_category>'.");
DEFINE_string(
    model_list, "",
    "A list of libsvm model files - one for each category. Each line is "
    "'<model_file>:<category>'.");
DEFINE_string(
    testing_list, "",
    "A list of pyramid files to classify using the models from model_list. "
    "Each line is '<test_file>:<ground_truth_category>'.");
DEFINE_string(
    result_file, "",
    "The file to write the result to.");
DEFINE_int32(
    thread_limit, 1,
    "The number of threads to use.");
DEFINE_string(
    kernel, "intersection",
    "The svm type. Options are \"intersection\" or \"linear\".");
using std::map;
using std::pair;
using std::set;
using std::string;
using std::vector;
using sjm::spatial_pyramid::SpatialPyramid;
using sjm::spatial_pyramid::SparseVectorFloat;

typedef map<string, SpatialPyramid> PyramidMap;
typedef map<string, svm_model*> SvmMap;
typedef map<string, pair<int, int> > ResultsMap;

enum SvmKernel {
  LINEAR_KERNEL,
  INTERSECTION_KERNEL
};

void Classify(const string test_filename,
              const string true_category,
              const PyramidMap& example_map,
              const SvmMap& svm_map,
              const SvmKernel svm_kernel,
              ResultsMap& results_map,
              boost::mutex& results_mutex) {
  svm_node* kernel_vector = new svm_node[example_map.size() + 2];
  int* label_vector = new int[2];
  double decision_value = 0;

  SpatialPyramid testing_pyramid;
  string pyramid_data;
  sjm::util::ReadFileToStringOrDie(test_filename, &pyramid_data);
  testing_pyramid.ParseFromString(pyramid_data);
  // Form the kernel vector for this example. The first index is zero
  // and the value doesn't matter. This is the libsvm library
  // convention for test vectors.
  kernel_vector[0].index = 0;
  kernel_vector[0].value = 0;
  int i = 1;
  for (PyramidMap::const_iterator it = example_map.begin();
       it != example_map.end(); ++it) {
    kernel_vector[i].index = i;
    if (svm_kernel == INTERSECTION_KERNEL) {
      kernel_vector[i].value =
          sjm::spatial_pyramid::SpmKernel(testing_pyramid, it->second,
                                          testing_pyramid.level_size());
    } else if (svm_kernel == LINEAR_KERNEL) {
      kernel_vector[i].value =
          sjm::spatial_pyramid::LinearKernel(testing_pyramid, it->second);
    }
    ++i;
  }
  kernel_vector[example_map.size() + 1].index = -1;
  kernel_vector[example_map.size() + 1].value = 0;

  // Now, get the decision values for the +1 class in each of the
  // models and find the category scored most strongly.
  string max_category = "";
  double max_score = -10000;
  for (SvmMap::const_iterator it = svm_map.begin();
       it != svm_map.end(); ++it) {
    const string category = it->first;
    const svm_model* model = it->second;
    // The meaning of decision_value depends on what label is in
    // position [0] of the labels structure.
    svm_get_labels(model, label_vector);
    svm_predict_values(model, kernel_vector, &decision_value);
    double category_score;
    if (label_vector[0] == 1) {
      // If the first label was +1, then the decision value is the
      // confidence for +1 class.
      category_score = decision_value;
    } else {
      // If the first label was not +1 (ie. it was -1), then the
      // decision value is the confidence for the -1 class.
      category_score = -decision_value;
    }
    if (category_score > max_score) {
      max_score = category_score;
      max_category = category;
    }
  }

  boost::mutex::scoped_lock l(results_mutex);
  if (results_map.find(true_category) != results_map.end()) {
    // We've already got some results for this category.
    int total = results_map[true_category].first;
    int correct = results_map[true_category].second;
    int new_total = total + 1;
    int new_correct = correct;
    if (max_category == true_category) {
      new_correct += 1;
    }
    results_map[true_category] = std::make_pair(new_total, new_correct);
  } else {
    if (max_category == true_category) {
      results_map[true_category] = std::make_pair(1, 1);
    } else {
      results_map[true_category] = std::make_pair(1, 0);
    }
  }

  LOG(INFO) << "File: " << test_filename <<
      ", Prediction: " << max_category;

  delete[] label_vector;
  delete[] kernel_vector;
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  CHECK(!FLAGS_result_file.empty()) << "--result_file needed.";

  // Parse the svm kernel parameter.
  SvmKernel svm_kernel;
  vector<string> svm_kernel_parts;
  if (FLAGS_kernel == "intersection") {
    svm_kernel = INTERSECTION_KERNEL;
  } else if (FLAGS_kernel == "linear") {
    svm_kernel = LINEAR_KERNEL;
  } else {
    LOG(FATAL) << "Unrecognized SVM kernel.";
  }

  // Load all the list data.
  string pyramid_list_data;
  sjm::util::ReadFileToStringOrDie(FLAGS_training_list, &pyramid_list_data);
  string model_list_data;
  sjm::util::ReadFileToStringOrDie(FLAGS_model_list, &model_list_data);
  string testing_list_data;
  sjm::util::ReadFileToStringOrDie(FLAGS_testing_list, &testing_list_data);
  vector<string> pyramid_list;
  boost::split(pyramid_list, pyramid_list_data, boost::is_any_of("\n"));
  vector<string> model_list;
  boost::split(model_list, model_list_data, boost::is_any_of("\n"));
  vector<string> testing_list;
  boost::split(testing_list, testing_list_data, boost::is_any_of("\n"));

  // Load the training data into a map sorted by filename.
  PyramidMap file_to_training_data;
  BOOST_FOREACH(string t, pyramid_list) {
    if (!t.empty()) {
      vector<string> training_parts;
      boost::split(training_parts, t, boost::is_any_of(":"));
      string training_pyramid_name = training_parts[0];
      SpatialPyramid pyramid;
      string pyramid_data;
      sjm::util::ReadFileToStringOrDie(training_pyramid_name, &pyramid_data);
      pyramid.ParseFromString(pyramid_data);
      file_to_training_data[t] = pyramid;
    }
  }

  // Load all the models and put in map, keyed by category name.
  SvmMap category_to_svm_model;
  BOOST_FOREACH(string t, model_list) {
    if (!t.empty()) {
      vector<string> model_parts;
      boost::split(model_parts, t, boost::is_any_of(":"));
      svm_model* model =
          svm_load_model(sjm::util::expand_user(model_parts[0]).c_str());
      category_to_svm_model[model_parts[1]] = model;
    }
  }

  ResultsMap results_map;
  boost::mutex results_mutex;
  vector<boost::thread*> threads_list;
  // Classify each of the test images and compare against the ground truth.
  BOOST_FOREACH(string t, testing_list) {
    if (!t.empty()) {
      vector<string> testing_parts;
      boost::split(testing_parts, t, boost::is_any_of(":"));
      string test_filename = testing_parts[0];
      string true_category = testing_parts[1];
      while (threads_list.size() >= FLAGS_thread_limit) {
        for (vector<boost::thread*>::iterator thread_it = threads_list.begin();
             thread_it != threads_list.end();
             ++thread_it) {
          if ((*thread_it)->timed_join(boost::posix_time::milliseconds(25))) {
            delete (*thread_it);
            threads_list.erase(thread_it);
            break;
          }
        }
      }
      // TODO(sanchom): Change results_map to a pointer instead of a
      // reference.  Same with results_mutex.
      boost::thread* classify_thread =
          new boost::thread(Classify, test_filename, true_category,
                            boost::ref(file_to_training_data),
                            boost::ref(category_to_svm_model),
                            svm_kernel,
                            boost::ref(results_map),
                            boost::ref(results_mutex));
      threads_list.push_back(classify_thread);
    }
  }
  // Join with any remaining threads.
  for (vector<boost::thread*>::iterator thread_it = threads_list.begin();
       thread_it != threads_list.end(); ++thread_it) {
    (*thread_it)->join();
    delete (*thread_it);
  }

  float average_accuracy = 0;
  for (ResultsMap::const_iterator it = results_map.begin();
       it != results_map.end(); ++it) {
    float category_accuracy =
        it->second.second / static_cast<float>(it->second.first);
    LOG(INFO) << "[" << it->first << "] accuracy: " << category_accuracy;
    average_accuracy += category_accuracy;
  }
  LOG(INFO) << "Mean accuracy: " << average_accuracy / results_map.size();
  const int kBufferSize = 50;
  char buffer[kBufferSize];
  std::snprintf(buffer, kBufferSize, "%f\n",
                average_accuracy / results_map.size());
  sjm::util::AppendStringToFileOrDie(FLAGS_result_file, buffer);

  return 0;
}
