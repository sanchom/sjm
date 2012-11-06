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

// This command line tool trains the SVM models for bag-of-words, SPM,
// or Spatially Local Coding classification.

#include <cstdlib>
#include <map>
#include <deque>
#include <set>
#include <string>
#include <vector>

#include "boost/lexical_cast.hpp"
#include "boost/thread.hpp"
#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "spatial_pyramid/spatial_pyramid.pb.h"
#include "spatial_pyramid/spatial_pyramid_kernel.h"
#include "svm/svm.h"
#include "util/util.h"

DEFINE_string(
    training_list, "",
    "A file listing all the training pyramid paths with "
    "their ground truth categories. Each line is <path>:<category>");
DEFINE_string(
    output_directory, "",
    "The directory at which the output model files will be saved.");
DEFINE_int32(
    thread_limit, 1,
    "The number of trainer threads.");
DEFINE_string(
    kernel, "intersection",
    "The svm type. Options are \"intersection\" or \"linear\".");
DEFINE_double(
    c, 0,
    "The regularizer for the SVM. If 0, the trainer will do cross "
    "validation to determine the appropriate setting.");
DEFINE_string(gram_matrix_checkpoint_file, "",
              "A file that is touched when gram matrix is completed.");
DEFINE_string(cross_validation_checkpoint_file, "",
              "A file that is touched when cross validation is completed.");

using std::make_pair;
using std::map;
using std::pair;
using std::deque;
using std::set;
using std::string;
using std::vector;
using sjm::spatial_pyramid::SpatialPyramid;
using sjm::spatial_pyramid::SparseVectorFloat;

typedef map<string, pair<SpatialPyramid, string> > TrainingExampleMap;

enum SvmKernel {
  LINEAR_KERNEL,
  INTERSECTION_KERNEL
};

void DoEnsembleCrossValidation(const TrainingExampleMap& training_examples,
                               const set<string>& category_set,
                               const int num_categories,
                               const SvmKernel svm_kernel,
                               const float c,
                               const int num_folds,
                               const svm_problem* problem,
                               map<float, float>* result_map,
                               boost::mutex* result_mutex) {
  // Split problem into (num_folds - 1) / num_folds training,
  // 1 / num_folds testing.
  // Eg. If num_folds = 3:
  // [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
  //  ^        ^        ^        ^            ^  Testing fold 1
  //     ^        ^        ^         ^           Testing fold 2
  //        ^        ^        ^          ^       Testing fold 3
  svm_parameter param;
  param.svm_type = C_SVC;
  param.kernel_type = PRECOMPUTED;
  param.C = c;
  param.coef0 = 0;
  param.degree = 3;
  param.gamma = 0;
  param.nr_weight = 1;
  param.weight_label = new int[1];
  param.weight_label[0] = -1;
  param.weight = new double[1];
  param.weight[0] = 1.0 / num_categories;
  param.shrinking = 0;
  param.cache_size = 4096;
  param.probability = 0;
  param.eps = 0.001;

  int num_correct = 0;
  int num_test = 0;
  for (int fold = 0; fold < num_folds; ++fold) {
    LOG(INFO) << "c = " << c << ", cross validation fold: " << fold;
    // Iterate through the problem, picking out rows to be training and testing.
    int i = 0;
    vector<string> true_training_categories;
    vector<string> true_testing_categories;
    vector<int> training_indices;
    vector<int> testing_indices;
    for (TrainingExampleMap::const_iterator it = training_examples.begin();
         it != training_examples.end(); ++it) {
      if ((i - fold) % num_folds == 0) {
        testing_indices.push_back(i);
        true_testing_categories.push_back(it->second.second);
      } else {
        training_indices.push_back(i);
        true_training_categories.push_back(it->second.second);
      }
      ++i;
    }
    // Extract subset of gram matrix associated with training examples.
    svm_problem subset_problem;
    subset_problem.l = training_indices.size();
    subset_problem.y = new double[subset_problem.l];
    subset_problem.x = new svm_node*[subset_problem.l];
    int row_index = 0;
    for (vector<int>::const_iterator a = training_indices.begin();
         a != training_indices.end(); ++a) {
      // Extract the appropriate columns from
      // this row.
      subset_problem.x[row_index] = new svm_node[subset_problem.l + 2];
      subset_problem.x[row_index][0].index = 0;
      subset_problem.x[row_index][0].value = row_index + 1;
      int col_index = 1;
      for (vector<int>::const_iterator b = training_indices.begin();
           b != training_indices.end(); ++b) {
        // Take the value from the a-th row, and the *b+1st colum.
        // We add 1 to b because the first element in the row is the id.
        subset_problem.x[row_index][col_index].index = col_index;
        subset_problem.x[row_index][col_index].value =
            problem->x[*a][*b + 1].value;
        ++col_index;
      }
      // The last item has index -1. The value is ignored.
      subset_problem.x[row_index][subset_problem.l + 1].index = -1;
      ++row_index;
    }

    // Train an SVM model for each class using the given C.
    map<string, svm_model*> ensemble;
    for (set<string>::const_iterator category = category_set.begin();
         category != category_set.end(); ++category) {
      // Set up the target vector to be +1 for this category, -1 otherwise.
      for (int i = 0; i < subset_problem.l; ++i) {
        if (true_training_categories[i] == *category) {
          subset_problem.y[i] = 1;
        } else {
          subset_problem.y[i] = -1;
        }
      }
      const char* check = svm_check_parameter(&subset_problem, &param);
      if (check != NULL) {
        LOG(INFO) << check;
      }
      svm_model* model = svm_train(&subset_problem, &param);
      ensemble[*category] = model;
    }

    // Get the predictive accuracy for this ensemble on the test instances.
    // For each test instance, build the kernel vector.
    svm_node* test_vector = NULL;
    int* label_vector = new int[2];
    double decision_value = 0;
    for (size_t i = 0; i < testing_indices.size(); ++i) {
      int testing_index = testing_indices[i];
      int vector_length = 0;
      // Problem size, +1 for index 0, +1 for final index.
      vector_length = subset_problem.l + 2;
      test_vector = new svm_node[vector_length];
      // This is where we look up the kernel values between the test
      // instance and the training instances.
      for (size_t j = 0; j < training_indices.size(); ++j) {
        int training_index = training_indices[j];
        test_vector[j + 1].index = j + 1;
        test_vector[j + 1].value =
            problem->x[testing_index][training_index + 1].value;
      }
      // The first index is zero, and its value doesn't matter for
      // test instances.
      test_vector[0].index = 0;
      test_vector[0].value = 0;
      // The last index is -1 by convension, and the value doesn't matter.
      test_vector[vector_length - 1].index = -1;
      test_vector[vector_length - 1].value = 0;

      // Now, get the predictions that the models give.
      string max_category = "";
      double max_score = -10000;
      for (map<string, svm_model*>::const_iterator model_it = ensemble.begin();
           model_it != ensemble.end(); ++model_it) {
        const string category = model_it->first;
        const svm_model* model = model_it->second;
        // The meaning of decision_value depends on what label is in
        // position [0].
        svm_get_labels(model, label_vector);
        svm_predict_values(model, test_vector, &decision_value);
        double category_score;
        if (label_vector[0] == 1) {
          category_score = decision_value;
        } else {
          category_score = -decision_value;
        }
        if (category_score > max_score) {
          max_score = category_score;
          max_category = category;
        }
      }
      ++num_test;
      if (max_category == true_testing_categories[i]) {
        ++num_correct;
      }
      delete[] test_vector;
      test_vector = NULL;
    }
    delete[] label_vector;

    for (map<string, svm_model*>::iterator it = ensemble.begin();
         it != ensemble.end(); ++it) {
      svm_free_and_destroy_model(&(it->second));
    }
    delete[] subset_problem.y;
    for (int i = 0; i < subset_problem.l; ++i) {
      delete[] subset_problem.x[i];
    }
    delete[] subset_problem.x;
  }
  LOG(INFO) << "Cross validation accuracy for c = " << c << ": " <<
      num_correct / static_cast<float>(num_test);
  if (result_mutex != NULL) {
    boost::mutex::scoped_lock l(*result_mutex);
    (*result_map)[c] = num_correct / static_cast<float>(num_test);
  } else {
    (*result_map)[c] = num_correct / static_cast<float>(num_test);
  }
}

void BuildGramMatrixShard(svm_node** x,
                          const int shard,
                          const SvmKernel svm_kernel,
                          const TrainingExampleMap& example_map) {
  int one_past_end = static_cast<int>(
      example_map.size() *
      (1 - std::sqrt(static_cast<float>(shard) / FLAGS_thread_limit)));
  int start_row = static_cast<int>(
      example_map.size() *
      (1 - std::sqrt(static_cast<float>(shard + 1) / FLAGS_thread_limit)));
  if (shard == FLAGS_thread_limit - 1) {
    start_row = 0;
  }
  if (one_past_end > static_cast<int>(example_map.size())) {
    one_past_end = example_map.size();
  }
  LOG(INFO) << "Shard " << shard << " is working on rows " <<
      start_row << " up to but not including row " <<
      one_past_end;
  int row_index = 0;
  for (TrainingExampleMap::const_iterator it_a = example_map.begin();
       it_a != example_map.end(); ++it_a) {
    if (row_index >= start_row && row_index < one_past_end) {
      LOG(INFO) << "[Shard " << shard << "]: Working on row " <<
          row_index << ".";
      // For precomputed kernels, each row needs data size + 2 columns.
      x[row_index] = new svm_node[example_map.size() + 2];
      // For pre-computed kernels, the first index (0) is the example
      // id (1-based).
      x[row_index][0].index = 0;
      x[row_index][0].value = row_index + 1;
      int j = 1;
      for (TrainingExampleMap::const_iterator it_b = example_map.begin();
           it_b != example_map.end(); ++it_b) {
        // We only build the upper triangle of the gram matrix,
        // including the diagonal.
        if (j >= row_index + 1) {
          x[row_index][j].index = j;
          if (svm_kernel == INTERSECTION_KERNEL) {
            x[row_index][j].value =
                sjm::spatial_pyramid::SpmKernel(
                    it_a->second.first,
                    it_b->second.first,
                    it_a->second.first.level_size());
          } else if (svm_kernel == LINEAR_KERNEL) {
            x[row_index][j].value =
                sjm::spatial_pyramid::LinearKernel(
                    it_a->second.first,
                    it_b->second.first);
          }
        }
        ++j;
      }
      // The last item has index -1. The value is ignored.
      x[row_index][example_map.size() + 1].index = -1;
    }
    ++row_index;
  }
}

bool StopCondition(const map<float, float>& result_map) {
  if (result_map.size() == 0) {
    return false;
  }
  if (result_map.rbegin()->first <= 32) {
    return false;
  }

  bool improvement = false;
  float prev_result = 2.0;
  int lookback_count = 0;
  for (map<float, float>::const_reverse_iterator result_it =
           result_map.rbegin();
       result_it != result_map.rend(); ++result_it) {
    if (lookback_count >= 5) {
      break;
    }
    LOG(INFO) << "Past result: c = " << result_it->first <<
        ", result = " << result_it->second;
    if (result_it->second < prev_result && prev_result < 1.5) {
      improvement = true;
    }
    prev_result = result_it->second;
    ++lookback_count;
  }
  return !improvement;
}

float KeyWithMaxValue(const map<float, float>& result_map) {
  CHECK_NE(result_map.size(), 0);
  float max_value = -1;
  float max_key = -1;
  for (map<float, float>::const_iterator it = result_map.begin();
       it != result_map.end(); ++it) {
    if (it->second > max_value) {
      max_value = it->second;
      max_key = it->first;
    }
  }
  return max_key;
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  CHECK(!FLAGS_training_list.empty()) << "--training_list is required.";
  CHECK(!FLAGS_output_directory.empty()) << "--output_basename is required.";

  const int kGeometricFolds = 5;
  const int kLinearFolds = 5;

  SvmKernel svm_kernel;
  vector<string> svm_kernel_parts;
  if (FLAGS_kernel == "intersection") {
    svm_kernel = INTERSECTION_KERNEL;
  } else if (FLAGS_kernel == "linear") {
    svm_kernel = LINEAR_KERNEL;
  } else {
    LOG(FATAL) << "Unrecognized SVM kernel.";
  }

  vector<boost::thread*> training_threads;
  vector<string> training_lines;
  sjm::util::ReadLinesFromFileIntoVectorOrDie(FLAGS_training_list,
                                              &training_lines);

  set<string> category_set;
  // Maps from an id (the path on disk) to a pair <pyramid, category>.
  TrainingExampleMap file_to_example;
  BOOST_FOREACH(string t, training_lines) {
    if (!t.empty()) {
      LOG(INFO) << "Loading " << t;
      vector<string> training_parts;
      boost::split(training_parts, t, boost::is_any_of(":"));
      string path = training_parts[0];
      string category = training_parts[1];
      SpatialPyramid pyramid;
      string pyramid_data;
      sjm::util::ReadFileToStringOrDie(path, &pyramid_data);
      pyramid.ParseFromString(pyramid_data);
      file_to_example[path] = make_pair(pyramid, category);
      category_set.insert(category);
    }
  }

  for (TrainingExampleMap::const_iterator it = file_to_example.begin();
       it != file_to_example.end(); ++it) {
    LOG(INFO) << it->first << ", levels: " << it->second.first.level_size() <<
        ", label: " << it->second.second;
  }

  LOG(INFO) << "Building the gram matrix.";
  svm_problem problem;
  problem.l = file_to_example.size();
  problem.y = new double[problem.l];
  problem.x = new svm_node*[problem.l];
  // Create the gram matrix once. Split into threads to fill up the
  // matrix in blocks.
  vector<boost::thread*> gram_threads;
  for (int shard = 0; shard < FLAGS_thread_limit; ++shard) {
    boost::thread* gram_thread =
        new boost::thread(
            BuildGramMatrixShard, problem.x, shard, svm_kernel,
            boost::ref(file_to_example));
    gram_threads.push_back(gram_thread);
  }
  // Join to all gram matrix threads.
  for (vector<boost::thread*>::iterator thread_it = gram_threads.begin();
       thread_it != gram_threads.end(); ++thread_it) {
    (*thread_it)->join();
    delete (*thread_it);
  }
  // Copy upper triangle of gram matrix into lower triangle.
  for (int row = 0; row < problem.l; ++row) {
    for (int col = 1; col < row + 1; ++col) {
      problem.x[row][col].index = col;
      problem.x[row][col].value = problem.x[col - 1][row + 1].value;
    }
  }

  if (!FLAGS_gram_matrix_checkpoint_file.empty()) {
    FILE* f = fopen(FLAGS_gram_matrix_checkpoint_file.c_str(), "w");
  }

  float selected_c = 0;
  if (FLAGS_c == 0) {
    // Do cross-validation on the training set.
    // First, do a geometric search.
    float c = 0.03125;
    bool done = false;
    map<float, float> result_map;
    boost::mutex result_mutex;
    vector<boost::thread*> thread_list;
    while (!done) {
      // If the thread limit has been reached, we wait to join and
      // delete with one of them.
      while (static_cast<int>(thread_list.size()) >= FLAGS_thread_limit) {
        for (vector<boost::thread*>::iterator thread_it = thread_list.begin();
             thread_it != thread_list.end();
             ++thread_it) {
          if ((*thread_it)->timed_join(boost::posix_time::milliseconds(1))) {
            delete (*thread_it);
            thread_list.erase(thread_it);
            break;
          }
        }
      }
      done = StopCondition(result_map);
      if (!done) {
        // Add a cross-validation thread and increment the c value for
        // the next one.
        boost::thread* t = new boost::thread(
            DoEnsembleCrossValidation,
            boost::ref(file_to_example),
            boost::ref(category_set),
            category_set.size(),
            svm_kernel,
            c,
            kGeometricFolds,
            &problem,
            &result_map,
            &result_mutex);
        thread_list.push_back(t);
        c *= 2;
      } else {
        // Wait for all remaining threads to finish.
        for (vector<boost::thread*>::iterator thread_it = thread_list.begin();
             thread_it != thread_list.end();
             ++thread_it) {
          (*thread_it)->join();
          delete (*thread_it);
        }
      }
    }
    // Then, do a finer search.
    float lower_bound = KeyWithMaxValue(result_map) / 2;
    float upper_bound = KeyWithMaxValue(result_map) * 2;
    float step = (upper_bound - lower_bound) / 10;
    result_map.clear();
    thread_list.clear();
    for (float c = lower_bound; c <= upper_bound; c += step) {
      // If the thread limit has been reached, we wait to join and
      // delete with one of them.
      while (static_cast<int>(thread_list.size()) >= FLAGS_thread_limit) {
        for (vector<boost::thread*>::iterator thread_it = thread_list.begin();
             thread_it != thread_list.end();
             ++thread_it) {
          if ((*thread_it)->timed_join(boost::posix_time::milliseconds(1))) {
            delete (*thread_it);
            thread_list.erase(thread_it);
            break;
          }
        }
      }
      boost::thread* t = new boost::thread(
          DoEnsembleCrossValidation,
          boost::ref(file_to_example),
          boost::ref(category_set),
          category_set.size(),
          svm_kernel,
          c,
          kLinearFolds,
          &problem,
          &result_map,
          &result_mutex);
      thread_list.push_back(t);
    }
    // Wait for all threads to be finished.
    for (vector<boost::thread*>::iterator thread_it = thread_list.begin();
         thread_it != thread_list.end();
         ++thread_it) {
      (*thread_it)->join();
      delete (*thread_it);
    }
    float best_c = KeyWithMaxValue(result_map);
    LOG(INFO) << "Selected c: " << best_c << ".";
    selected_c = best_c;
  } else {
    selected_c = FLAGS_c;
  }

  if (!FLAGS_cross_validation_checkpoint_file.empty()) {
    FILE* f = fopen(FLAGS_cross_validation_checkpoint_file.c_str(), "w");
  }

  svm_parameter param;
  param.svm_type = C_SVC;
  param.kernel_type = PRECOMPUTED;
  param.C = selected_c;
  param.coef0 = 0;
  param.degree = 3;
  param.gamma = 0;
  param.nr_weight = 1;
  param.weight_label = new int[1];
  param.weight_label[0] = -1;
  param.weight = new double[1];
  param.weight[0] = 1.0 / category_set.size();
  param.shrinking = 0;
  param.cache_size = 4096;
  param.probability = 0;
  param.eps = 0.001;

  for (set<string>::const_iterator category = category_set.begin();
       category != category_set.end(); ++category) {
      int i = 0;
      for (TrainingExampleMap::const_iterator training_example_it =
               file_to_example.begin();
           training_example_it != file_to_example.end();
           ++training_example_it) {
        if (training_example_it->second.second == *category) {
          problem.y[i] = 1;
        } else {
          problem.y[i] = -1;
        }
        ++i;
      }
      svm_model* model = svm_train(&problem, &param);

      boost::filesystem::path output_directory(FLAGS_output_directory);
      boost::filesystem::path filename(*category + ".svm");
      boost::filesystem::path full_output_path = output_directory / filename;
      CHECK_EQ(0,
               svm_save_model(
                   full_output_path.string().c_str(),
                   model)) << "Error saving " << full_output_path;
      LOG(INFO) << "[" << *category << "] Saved model.";
      svm_free_model_content(model);
      svm_free_and_destroy_model(&model);
  }

  delete[] param.weight;
  delete[] param.weight_label;
  delete[] problem.y;
  for (size_t i = 0; i < file_to_example.size(); ++i) {
    delete[] problem.x[i];
  }
  delete[] problem.x;

  return 0;
}
