LICENCE
=======
Copyright (c) 2010-2012, Sancho McCann

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

- Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

DOCKER
======
The dockerfile describes how to set up an environment that will let you
compile and run this code.

An image is also available at [sanchom/phd-environment](https://hub.docker.com/r/sanchom/phd-environment/).

This is an example of how you could use it.

    git clone https://github.com/sanchom/sjm.git
    cd sjm
    docker pull sanchom/phd-environment
    docker run --rm -v `pwd`:`pwd` -w `pwd` sanchom/phd-environment scons

This puts the repository on your local machine. Then, it grabs the docker image that has the build
environment all set up. The `docker run` command does a few things. By using ``-v `pwd`:`pwd` ``,
docker attaches the current directory (the `sjm` repository) to an identical path inside the
container. ``-w `pwd` `` makes the container start with that path as the working directory. `scons` is
the command that builds everything. It runs inside the container, but produces binaries in that directory
that was just attached with `-v`, so the build output will be visible to you on your host machine and
persistant across container launches.

USING
=====

NBNN and Local NBNN Experiments
-------------------------------
This section explains how to run the experiments described in
[_Local Naive Bayes Nearest Neighbor for Image Classification_]().

### Preparing the image dataset

1. Download the [Caltech 101 Dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech101/).
2. Extract the files: `tar -xvzf 101_ObjectCategories.tar.gz`. This should give you a directory called
`101_ObjectCategories` with 102 sub-directories, one for each of the 101 object categories and one background
category.
3. Resize the images to have a maximum width or height of 300 pixels, with preserved aspect ratio. To do this,
I use ImageMagick's `mogrify` command: `mogrify -verbose -resize 300x300 101_ObjectCategories/*/*.jpg`. According
to the ImageMagick documentation, this uses a Mitchell filter if an image is enlarged to 300x300, or a Lanczos
filter if the image is shrunk to 300x300.

### Extracting the SIFT features

I wrote a command line tool that you installed above called `extract_descriptors_cli`, but the Python wrapper around
that is more convenient to use. Change directories to `sjm/naive_bayes_nearest_neighbor/experiment_1` and run this:

    python extract_caltech.py --dataset_path=[path_to_your_101_Categories_directory] \
    --process_limit [num_processes_to_spawn] --sift_normalization_threshold 2.0 --sift_discard_unnormalized \
    --sift_grid_type FIXED_3X3 --sift_first_level_smoothing 0.66 --sift_fast --sift_multiscale \
    --features_directory [path_for_extracted_features]

This will extract multi-scale SIFT at 4 scales, with a small amount of additional smoothing applied at the first level.
Features are extracted at each level on a 3x3 pixel grid. The first level features are 16x16, and increase by a factor
of 1.5 at each level. This also discards features that from low contrast regions
(`--sift_normalization_threshold 2.0 --sift_discard_unnormalized`).

Now, you should have a directory structure at `path_for_extracted_features` that mirrors that at
`path_to_your_101_Categories`, but with `.sift` files instead of `.jpeg` files.

### Running standard NBNN

The code to train and test standard NBNN is in `sjm/naive_bayes_nearest_neighbor/experiment_1`.

1. Change to the `sjm/naive_bayes_nearest_neighbor/experiment_1` directory.
2. Create a `101_categories.txt` file that lists all the 101 object categories (not BACKGROUND_Google). We ignore
the background class as suggested by the dataset creators:
http://authors.library.caltech.edu/7694/1/CNS-TR-2007-001.pdf
3. Run this:

    ./experiment_1 --category_list 101_categories.txt
    --features_directory [path_for_extracted_features]
    --alpha [alpha] --trees [trees] --checks [checks]
    --results_file [results_file] --logtostderr

In our experiments, we fixed alpha=1.6, trees=4, and varied the checks variable depending on the particular experiment
we were performing, but for optimal performance, checks should be greater than 128 (see Figure 4 from our paper).

### Running Local NBNN

The code to train and test local NBNN is in `sjm/naive_bayes_nearest_neighbor/experiment_3`.

1. Change to the `sjm/naive_bayes_nearest_neighbor/experiment_1` directory.
2. Create a `101_categories.txt` file that lists all the 101 object categories (not BACKGROUND_Google). We ignore
the background class as suggested by the dataset creators:
http://authors.library.caltech.edu/7694/1/CNS-TR-2007-001.pdf
3. Run this:

    ./experiment_3 --category_list 101_categories.txt
    --features_directory [path_for_extracted_features]
    --alpha [alpha] --trees [trees] --checks [checks]
    --k [k]
    --num_test [num_test] --num_train [num_train]
    --results_file [results_file] --logtostderr

In our experiments, we fixed alpha=1.6, trees=4, and varied k and checks depending on the experiment.
For optimal results, checks should be above 1024 (see Figure 4 from our paper), and k should be around 10-20
(see Figure 3 from our paper).

Inspecting the algorithms
=========================
The NBNN algorithm is implemented in [`NbnnClassifier::Classify`](https://github.com/sanchom/sjm/blob/master/naive_bayes_nearest_neighbor/nbnn_classifier-inl.h#L92)

The Local NBNN algorithm is implemented in [`MergedClassifier::Classify`](https://github.com/sanchom/sjm/blob/master/naive_bayes_nearest_neighbor/merged_classifier.h#L166)

Spatially Local Coding experiments
----------------------------------
This section explains how to run the experiments described in
_Spatially Local Coding for Object Recognition_.

### Preparing the image dataset

(This is the same process as for the NBNN and Local NBNN experiments above.)

1. Download the [Caltech 101 Dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech101/).
2. Extract the files: `tar -xvzf 101_ObjectCategories.tar.gz`. This should give you a directory called
`101_ObjectCategories` with 102 sub-directories, one for each of the 101 object categories and one background
category.
3. Resize the images to have a maximum width or height of 300 pixels, with preserved aspect ratio. To do this,
I use ImageMagick's `mogrify` command: `mogrify -verbose -resize 300x300 101_ObjectCategories/*/*.jpg`. According
to the ImageMagick documentation, this uses a Mitchell filter if an image is enlarged to 300x300, or a Lanczos
filter if the image is shrunk to 300x300.

### Running Localized Soft Assignment SPM

Assume that your Caltech directory is at [caltechdir], you are storing extracted features in [featuredir]
and results in [resultdir].

    rm -f /tmp/*.svm; rm -rf [featuredir]; \
    python baseline_experiment.py --dataset_path=[caltechdir] --work_directory=[resultdir] --clobber \
    --sift_normalization_threshold=2.0 --sift_discard_unnormalized \
    --sift_grid_type=FIXED_8X8 --sift_first_level_smoothing=0.66 --features_directory=[featuredir] \
    --process_limit=[num_processes] --num_train=[num_train] --num_test=15 --codeword_locality=10 --pooling=MAX_POOLING \
    --dictionary_training_size=1000000 --clobber_dictionary --pyramid_levels=3 --kmeans_accuracy=0.9 \
    --dictionary=[dictionary_size]:0 --kernel=[intersection|linear]

What makes this Localized Soft Assignment SPM is: `--codeword_locality=10 --pooling=MAX_POOLING --pyramid_levels=3`.

### Running Standard SPM

    rm -f /tmp/*.svm; rm -rf [featuredir]; \
    python baseline_experiment.py --dataset_path=[caltechdir] --work_directory=[resultdir] --clobber \
    --sift_normalization_threshold=2.0 --sift_discard_unnormalized \
    --sift_grid_type=FIXED_8X8 --sift_first_level_smoothing=0.66 --features_directory=[featuredir] \
    --process_limit=[num_processes] --num_train=[num_train] --num_test=15 --codeword_locality=1 --pooling=AVERAGE_POOLING \
    --dictionary_training_size=1000000 --clobber_dictionary --pyramid_levels=3 --kmeans_accuracy=0.9 \
    --dictionary=[dictionary_size]:0 --kernel=[intersection|linear]

What makes this Standard SPM is: `--codeword_locality=1 --pooling=AVERAGE_POOLING --pyramid_levels=3`.

### Running Spatially Local Coding

    rm -f /tmp/*.svm; rm -rf [featuredir]; \
    python baseline_experiment.py --dataset_path=[caltechdir] --work_directory=[resultdir] --clobber \
    --sift_normalization_threshold=2.0 --sift_discard_unnormalized \
    --sift_grid_type=FIXED_8X8 --sift_first_level_smoothing=0.66 --features_directory=[featuredir] \
    --process_limit=[num_processes] --num_train=[num_train] --num_test=15 --codeword_locality=10 --pooling=MAX_POOLING \
    --dictionary_training_size=1000000 --clobber_dictionary --pyramid_levels=1 --kmeans_accuracy=0.9 \
    --dictionary=[dictionary_size]:0 --dictionary=[dictionary_size]:0.75 \
    --dictionary=[dictionary_size]:1.5 --dictionary=[dictionary_size]:3.00 --kernel=linear

What makes this Spatially Local Coding is: `--codeword_locality=10`, `--pooling=MAX_POOLING`, `--pyramid_levels=1`,
and `--dictionary=[dictionary_size]:0`, `--dictionary=[dictionary_size]:0.75`,
`--dictionary=[dictionary_size]:1.5`, `--dictionary=[dictionary_size]:3.00`. The values after [dictionary_size] are
the location weightings to use for each dictionary.

### Changing the extraction settings

If you are using Caltech 256, pass the `--caltech256` flag.

You can use other `--sift_grid_type`s to get different extraction densities.
FIXED_8X8 without the `--sift_multiscale` flag extracts singlescale
SIFT every 8 pixels. If you use SCALED_BIN_WIDTH without the `--sift_multiscale` flag, you'll get singlescale
SIFT every 4 pixels. If you use SCALED_DOUBLE_BIN_WIDTH _with_ `--sift_multiscale`, you'll get 3 scales of SIFT,
with the lowest scale being every 8 pixels. If you use SCALED_BIN_WIDTH with `--sift_multiscale`, you'll get 3
scales of SIFT with the lowest scale being every 4 pixels.

### Inspecting the algorithm

There isn't a single place to point you to if you'd like to inspect the algorithm. Spatially Local Coding
is simply building multiple codebooks, each taking location into account to a different degree.
Then instead of building a spatial pyramid, just build a bag-of-words histogram for each of those codebooks and
concatenate them. That is the model for an image.

The approximate k-means clustering is implemented at: [`CodebookBuilder::ClusterApproximately`](https://github.com/sanchom/sjm/blob/master/codebooks/codebook_builder.cc#L170)

Spatial pyramid construction, including the option for coding across more than a single dictionary is implemented at: [`SpatialPyramidBuilder::BuildPyramid`](https://github.com/sanchom/sjm/blob/master/spatial_pyramid/spatial_pyramid_builder.cc#L132)

The SPM kernels can be inspected at [`spatial_pyramid_kernel.cc`](https://github.com/sanchom/sjm/blob/master/spatial_pyramid/spatial_pyramid_kernel.cc).

Our trainer wrapper for extensive cross-validation and one-vs-all SVM training is at [`trainer_cli.cc`](https://github.com/sanchom/sjm/blob/master/spatial_pyramid/trainer_cli.cc).