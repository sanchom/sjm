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

#include "codebooks/codebook_builder.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <random>
#include <set>
#include <vector>

#include "flann/flann.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "codebooks/dictionary.pb.h"
#include "sift/sift_descriptors.pb.h"

using std::copy;
using std::set;
using std::vector;

typedef std::mt19937 RandomNumberGenerator;  // Mersenne Twister

DEFINE_string(initialization_checkpoint_file, "",
              "A file that is touched when initialization is complete.");

namespace sjm {
namespace codebooks {

void CodebookBuilder::AddData(const sjm::sift::DescriptorSet& descriptors,
                              const float percentage,
                              const float location_weighting) {
  if (descriptors.sift_descriptor_size() > 0) {
    // Getting the descriptor dimensionality.
    data_dimensions_ = descriptors.sift_descriptor(0).bin_size();
    // If we're appending location, the dimensionality increases by 2.
    if (location_weighting > 0) {
      data_dimensions_ += 2;
    }
  } else {
    // There was no data to add.
    return;
  }

  if (!data_) {
    // On the initial AddData, we just allocate enough storage for
    // 100% of the descriptors.
    data_ = new flann::Matrix<float>(
        new float[descriptors.sift_descriptor_size() * data_dimensions_],
        descriptors.sift_descriptor_size(), data_dimensions_);
  } else {
    // On subsequent additions, we double the size of the storage
    // until there's enough storage for the 100% of the additional
    // data.
    size_t required_data_size =
        matrix_usage_ + descriptors.sift_descriptor_size();
    // TODO(sanchom): Handle the case where doubling the memory
    // allocation would fail, and adaptively back-off the requested
    // amount by calling new(nothrow).
    while (data_->rows < required_data_size) {
      int new_size = data_->rows * 2;
      flann::Matrix<float>* larger_data =
          new flann::Matrix<float>(new float[new_size * data_dimensions_],
                                   new_size, data_dimensions_);
      // Copy old data into larger data space.
      copy((*data_)[0],
           (*data_)[data_->rows - 1] + data_->cols,
           (*larger_data)[0]);
      // Delete smaller data structure.
      delete[] data_->ptr();
      delete data_;
      // Re-assign pointer to point to new, larger data structure.
      data_ = larger_data;
    }
  }

  // Putting a percentage of the descriptors into data_.
  for (int i = 0; i < descriptors.sift_descriptor_size(); ++i) {
    // TODO(sanchom): Replace with new C++11 <random> library usage.
    if (std::rand() / static_cast<float>(RAND_MAX) < percentage) {
      int next_row = matrix_usage_;  // This is the next row of data_
                                     // to use.
      // Copying the data from the descriptor into the matrix row.
      for (int d = 0; d < descriptors.sift_descriptor(i).bin_size(); ++d) {
        (*data_)[next_row][d] = descriptors.sift_descriptor(i).bin(d);
      }
      // Copying the location.
      if (location_weighting > 0) {
        // We multiply by 127 because the SIFT descriptor has been
        // normalized to be of length 127 rather than 1.0 so that it
        // can be stored in uint8s in the protocol buffer. We need to
        // multiply the x and y by 127 (they're stored in [0,1]) so
        // that the location weighting is more interpretable.
        (*data_)[next_row][data_dimensions_ - 2] =
            (descriptors.sift_descriptor(i).x() * 127 * location_weighting);
        (*data_)[next_row][data_dimensions_ - 1] =
            (descriptors.sift_descriptor(i).y() * 127 * location_weighting);
      }
      ++matrix_usage_;
    }
  }
}

// TODO(sanchom): Calling this when no data has been added gives a
// segmentation fault. Fix this.
void CodebookBuilder::Cluster(const int num_clusters_requested,
                              const int num_iterations) {
  CHECK_GT(num_clusters_requested, 0) << "Num clusters must be greater than 0";
  // First, truncate the data_ to the actual usage. This unfortunately
  // requires allocating additional memory, just to get rid of memory.
  // TODO(sanchom): Look into using realloc instead.
  flann::Matrix<float>* truncated_matrix =
      new flann::Matrix<float>(new float[matrix_usage_ * data_dimensions_],
                               matrix_usage_, data_dimensions_);
  // Copy data into smaller data space.
  copy((*data_)[0],
       (*data_)[matrix_usage_ - 1] + data_->cols,
       (*truncated_matrix)[0]);
  // Delete original, larger data structure.
  delete[] data_->ptr();
  delete data_;
  // Re-assign pointer to new, smaller data structure.
  data_ = truncated_matrix;

  // Do the kmeans clustering using flann's kmeans index with
  // num_clusters clusters, num_iterations, kmeans++ initialization.
  const flann::KMeansIndexParams index_params(
      num_clusters_requested, num_iterations, flann::CENTERS_KMEANSPP);
  flann::KMeansIndex<flann::L2<float> > index(*data_, index_params);
  index.buildIndex();

  // Get the centroids out of the kmeans index.
  if (centroids_) {
    delete[] centroids_->ptr();
    delete centroids_;
  }
  centroids_ = new flann::Matrix<float>(
      new float[num_clusters_requested * data_dimensions_],
      num_clusters_requested, data_dimensions_);
  int num_clusters = index.getClusterCenters(*centroids_);
  CHECK_EQ(num_clusters_requested, num_clusters) <<
      "Didn't build requested number of clusters.";
}

void CodebookBuilder::ClusterApproximately(
    const int num_clusters_requested,
    const int num_iterations,
    const float accuracy,
    const KMeansInitialization initialization,
    double* metric,
    vector<int>* sizes) {
  if (accuracy == 1.0) {
    return Cluster(num_clusters_requested, num_iterations);
  } else {
    CHECK_GT(num_clusters_requested, 0) <<
        "Num clusters must be greater than 0";
    // First, truncate the data to the actual usage. This
    // unfortunately requires allocating additional memory, just to
    // get rid of memory.
    // TODO(sanchom): Look into using realloc instead.
    flann::Matrix<float>* truncated_matrix =
        new flann::Matrix<float>(new float[matrix_usage_ * data_dimensions_],
                                 matrix_usage_, data_dimensions_);
    // Copy data into smaller data space.
    copy((*data_)[0],
         (*data_)[matrix_usage_ - 1] + data_->cols,
         (*truncated_matrix)[0]);
    // Delete original, larger data structure.
    delete[] data_->ptr();
    delete data_;
    // Re-assign pointer to new, smaller data structure.
    data_ = truncated_matrix;

    // KMeanPP initialization.
    LOG(INFO) << "Doing kmeanspp initialization.";
    const int n = data_->rows;
    RandomNumberGenerator random_number_generator;
    random_number_generator.seed(std::time(NULL));
    if (centroids_) {
      delete[] centroids_->ptr();
      delete centroids_;
    }
    centroids_ =
        new flann::Matrix<float>(
            new float[num_clusters_requested * data_dimensions_],
            num_clusters_requested, data_dimensions_);
    if (initialization == sjm::codebooks::SUBSAMPLED_KMEANS_PP ||
        initialization == sjm::codebooks::KMEANS_PP) {
      // Subsample 10% of the original data from which to select the
      // initial clusters.
      //
      // TODO(sanchom): Make the subsample percentage a parameter.
      int subsampled_n = 0;
      if (initialization == sjm::codebooks::SUBSAMPLED_KMEANS_PP) {
        subsampled_n = static_cast<int>(n * 0.1 + 0.5);
      } else {
        subsampled_n = n;
      }
      std::unique_ptr<double[]> distance_weighting(new double[subsampled_n]);
      vector<uint32_t> subsampled_rows;
      if (initialization == sjm::codebooks::SUBSAMPLED_KMEANS_PP) {
        set<uint32_t> subsampled_rows_set;
        std::uniform_int_distribution<uint32_t> random_n(0, n - 1);
        while (subsampled_rows_set.size() < subsampled_n) {
          uint32_t index = random_n(random_number_generator);
          if (subsampled_rows_set.count(index) == 0) {
            subsampled_rows_set.insert(index);
          }
        }
        // Now, move the subsampled row_ids into a vector.
        subsampled_rows.resize(subsampled_n);
        copy(subsampled_rows_set.begin(), subsampled_rows_set.end(),
             subsampled_rows.begin());
      } else {
        for (int i = 0; i < n; ++i) {
          subsampled_rows.push_back(i);
        }
      }

      // Get a random distribution over the subsampled size.
      std::uniform_int_distribution<uint32_t>
          random_subsampled_n(0, subsampled_n - 1);
      // 1. Choose an initial center uniformly at random from the data.
      uint32_t index =
          subsampled_rows[random_subsampled_n(random_number_generator)];
      CHECK(index >= 0 && index < n);
      // Copies the descriptor from (*data_)[index] into (*centroids_)[0].
      copy((*data_)[index], (*data_)[index] + data_->cols,
           (*centroids_)[0]);
      LOG_EVERY_N(INFO, 100) << "Placed center " << 0;

      double weight_total = 0;
      // Compute the initial distance weightings.
      //
      // i is an index into the subsampled row set
      for (int i = 0; i < subsampled_n; ++i) {
        double distance_squared = 0;
        for (int d = 0; d < data_dimensions_; ++d) {
          double diff = (*data_)[subsampled_rows[i]][d] - (*centroids_)[0][d];
          distance_squared += diff * diff;
        }
        distance_weighting[i] = distance_squared;
        weight_total += distance_squared;
      }

      // For all remaining centers to be allocated.
      for (int center_id = 1; center_id < num_clusters_requested; ++center_id) {
        // Select the next center according to the probabilities.
        double random_value =
            std::uniform_real_distribution<double>(0, weight_total)(
                random_number_generator);
        int next_center = -1;
        for (int index = 0; index < subsampled_n; ++index) {
          if (random_value <= distance_weighting[index]) {
            next_center = subsampled_rows[index];
            break;
          } else {
            random_value -= distance_weighting[index];
          }
        }

        CHECK_GE(next_center, 0);
        copy((*data_)[next_center], (*data_)[next_center] + data_->cols,
             (*centroids_)[center_id]);
        LOG_EVERY_N(INFO, 100) << "Placed center " << center_id;

        weight_total = 0;
        // Adjust the distance weightings and weight total.
        // Checking to see if there's a new minimum for each point.
        for (int i = 0; i < subsampled_n; ++i) {
          double distance_squared = 0;
          for (int d = 0; d < data_dimensions_; ++d) {
            double diff =
                (*data_)[subsampled_rows[i]][d] - (*centroids_)[center_id][d];
            distance_squared += diff * diff;
            if (distance_squared > distance_weighting[i]) {
              // If we're already more than the current distance
              // weighting, break.
              break;
            }
          }
          if (distance_squared < distance_weighting[i]) {
            distance_weighting[i] = distance_squared;
          }
          weight_total += distance_weighting[i];
        }
      }
    } else if (initialization == sjm::codebooks::KMEANS_RANDOM) {
      set<uint32_t> selected_centers;
      // This distribution will choose an int randomly from [0, n - 1].
      std::uniform_int_distribution<uint32_t> random_n(0, n - 1);
      for (int center_id = 0; center_id < num_clusters_requested; ++center_id) {
        // Find a new center that hasn't been selected yet.
        uint32_t selected_index = random_n(random_number_generator);
        while (selected_centers.count(selected_index) != 0) {
          selected_index = random_n(random_number_generator);
        }
        copy((*data_)[selected_index], (*data_)[selected_index] + data_->cols,
             (*centroids_)[center_id]);
        LOG_EVERY_N(INFO, 100) << "Placed center " << center_id;
      }
    } else {
      LOG(FATAL) << "Unhandled k-means initialization method.";
    }

    // Initialization is complete.

    // This just touches a checkpoint file if
    // FLAGS_initialization_checkpoint_file was set through the Google
    // gflags command line flags interface. I use this for monitoring
    // timing of various phases of execution.
    if (!FLAGS_initialization_checkpoint_file.empty()) {
      FILE* f = fopen(FLAGS_initialization_checkpoint_file.c_str(), "w");
    }

    flann::flann_algorithm_t autotune_selected_algorithm =
        flann::FLANN_INDEX_LINEAR;
    flann::KMeansIndexParams autotune_selected_kmeans_index_params;
    flann::KDTreeIndexParams autotune_selected_kdtree_index_params;
    flann::SearchParams autotune_selected_search_params;
    bool autotuning_complete = false;
    // This is where the approximate k-means is implemented. We do
    // approximate matching of points to centroids at each
    // iteration. For the approximate matching, we use an autotuned
    // FLANN index. However, we don't need to autotune each
    // iteration. We can use the autotuned parameters from the first
    // iteration and re-use them on subsequent iterations. The dataset
    // characteristics do not change significantly, so the autotuning
    // from the first iteration gives approximately the same accuracy
    // on subsequent iterations.
    for (int iteration = 0; iteration < num_iterations; ++iteration) {
      LOG(INFO) << "K-means iteration " << iteration;
      // Create an approximate index out of the centroids.
      std::unique_ptr<flann::Index<flann::L2<float> > > centroid_index;
      if (!autotuning_complete) {
        const float kBuildWeight = 0;
        const float kMemoryWeight = 0;
        const float kSampleFraction = 1;
        // Do autotuning.
        const flann::AutotunedIndexParams params(
            accuracy, kBuildWeight, kMemoryWeight, kSampleFraction);
        centroid_index.reset(
            new flann::Index<flann::L2<float> >(*centroids_, params));
        LOG(INFO) << "Building autotuned centroid index.";
        centroid_index->buildIndex();
        autotuning_complete = true;
        // Save the algorithm type.
        autotune_selected_algorithm =
            flann::get_param(centroid_index->getIndex()->getParameters(),
                             "algorithm", flann::FLANN_INDEX_LINEAR);
        LOG(INFO) << "Selected algorithm: " << autotune_selected_algorithm;
        autotune_selected_search_params =
            dynamic_cast<flann::AutotunedIndex<flann::L2<float> >* >(
                centroid_index->getIndex())->getSearchParameters();
        if (autotune_selected_algorithm == flann::FLANN_INDEX_LINEAR) {
          // Nothing to save.
        } else if (autotune_selected_algorithm == flann::FLANN_INDEX_KDTREE) {
          autotune_selected_kdtree_index_params =
              flann::KDTreeIndexParams(
                  flann::get_param(centroid_index->getIndex()->getParameters(),
                                   "trees", 1));
        } else if (autotune_selected_algorithm == flann::FLANN_INDEX_KMEANS) {
          flann::IndexParams params =
              centroid_index->getIndex()->getParameters();
          autotune_selected_kmeans_index_params =
              flann::KMeansIndexParams(
                  flann::get_param(params, "branching", 32),
                  flann::get_param(params, "iterations", 11),
                  flann::get_param(params, "centers_init",
                                   flann::FLANN_CENTERS_RANDOM),
                  flann::get_param(params, "cb_index", 0.2));
        } else {
          LOG(FATAL) << "Autotuning selected unhandled index type.";
        }
      } else {
        flann::IndexParams params;
        if (autotune_selected_algorithm == flann::FLANN_INDEX_LINEAR) {
          params = flann::LinearIndexParams();
        } else if (autotune_selected_algorithm == flann::FLANN_INDEX_KMEANS) {
          params = autotune_selected_kmeans_index_params;
        } else if (autotune_selected_algorithm == flann::FLANN_INDEX_KDTREE) {
          params = autotune_selected_kdtree_index_params;
        }
        centroid_index.reset(
            new flann::Index<flann::L2<float> >(*centroids_, params));
        LOG(INFO) << "Building centroid index with saved parameters.";
        centroid_index->buildIndex();
      }
      LOG(INFO) << "Matching points to centroids.";

      // Find nearest neighbors of data in the centroids.
      // The '1' is for finding a single nearest neighbor.
      flann::Matrix<int> indices(new int[data_->rows], data_->rows, 1);
      flann::Matrix<float> dists(new float[data_->rows], data_->rows, 1);
      centroid_index->knnSearch(
          *data_, indices, dists, 1,
          autotune_selected_search_params.checks);

      // Find mean of each centroid's assigned data.
      flann::Matrix<float> new_centroids(
          new float[centroids_->rows * centroids_->cols],
          centroids_->rows, centroids_->cols);
      std::memset(new_centroids.ptr(), 0,
                  new_centroids.rows * new_centroids.cols * sizeof(float));
      std::vector<int> centroid_sizes(centroids_->rows, 0);
      for (int i = 0; i < n; ++i) {
        int assigned_centroid = indices[i][0];
        // Assign this data item to the centroid.
        for (int d = 0; d < data_dimensions_; ++d) {
          new_centroids[assigned_centroid][d] +=
              (*data_)[i][d];
        }
        centroid_sizes[assigned_centroid] += 1;
      }

      int empty_cluster_count = 0;
      // Move each centroid to its mean.
      for (int i = 0; i < centroids_->rows; ++i) {
        // Divide each dimension by the total number of data items
        // assigned to the new centroid.
        for (int d = 0; d < data_dimensions_; ++d) {
          if (centroid_sizes[i] != 0) {
            (*centroids_)[i][d] = new_centroids[i][d] / centroid_sizes[i];
          } else {
            ++empty_cluster_count;
          }
        }
      }
      // This is just for diagnositcs. There should be no empty
      // clusters.
      LOG(INFO) << "Empty clusters: " << empty_cluster_count;

      // TODO(sanchom): Handle degenerate cases (split up cluster with
      // largest variance). For every centroid with zero members,
      // reset it to be a random member of a large variance cluster.

      // If this was the last iteration, we'll optionally compute a
      // metric for the clustering.
      if (metric != NULL && iteration == num_iterations - 1) {
        LOG(INFO) << "Computing the k-means metric.";
        *metric = 0;
        flann::IndexParams params;
        if (autotune_selected_algorithm == flann::FLANN_INDEX_LINEAR) {
          params = flann::LinearIndexParams();
        } else if (autotune_selected_algorithm == flann::FLANN_INDEX_KMEANS) {
          params = autotune_selected_kmeans_index_params;
        } else if (autotune_selected_algorithm == flann::FLANN_INDEX_KDTREE) {
          params = autotune_selected_kdtree_index_params;
        }
        centroid_index.reset(
            new flann::Index<flann::L2<float> >(*centroids_, params));
        LOG(INFO) << "Building centroid index with saved parameters.";
        centroid_index->buildIndex();
        LOG(INFO) << "Doing the search.";
        centroid_index->knnSearch(
            *data_, indices, dists, 1,
            autotune_selected_search_params.checks);
        LOG(INFO) << "Computing the stats.";
        for (int i = 0; i < n; ++i) {
          if (metric != NULL) {
            *metric += dists[i][0];
          }
        }
      }

      // If this was the last iteration, we'll optionally return the
      // cluster cardinalities.
      if (sizes != NULL && iteration == num_iterations - 1) {
        sizes->resize(centroid_sizes.size());
        std::copy(centroid_sizes.begin(), centroid_sizes.end(), sizes->begin());
        std::sort(sizes->begin(), sizes->end());
        std::reverse(sizes->begin(), sizes->end());
      }

      delete[] new_centroids.ptr();
      delete[] indices.ptr();
      delete[] dists.ptr();
    }
  }
}

void CodebookBuilder::GetDictionary(Dictionary* dictionary) const {
  // This converts the results from the clustering (centroids_) into a
  // Dictionary protocol buffer.
  dictionary->Clear();
  if (centroids_ != NULL) {
    for (size_t i = 0; i < centroids_->rows; ++i) {
      Centroid* c = dictionary->add_centroid();
      // Add as many bins as there are data dimensions.
      for (size_t j = 0; j < centroids_->cols; ++j) {
        c->add_bin((*centroids_)[i][j]);
      }
    }
  }
}

int CodebookBuilder::DataSize() const {
  return matrix_usage_;
}

}}  // namespace.
