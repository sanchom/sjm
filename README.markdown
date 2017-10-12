# Compiling

The recommended build process involves using
[docker](https://www.docker.com/), so that you have the exact
environment that I know this code works under.

The docker image can be built using the Dockerfile in this
repository. It is also available pre-built at
[sanchom/phd-environment](https://hub.docker.com/r/sanchom/phd-environment/).

I recommend checking out this repository into a location on your host
machine. Then launch the docker container as needed, attaching the
local directory to the container using docker's `-v` flag.

    git clone https://github.com/sanchom/sjm.git
    cd sjm
    docker run --rm -v `pwd`:/work -w /work sanchom/phd-environment scons

In case you're not familiar with docker, this command makes the local
directory visible inside the container at the path `/work`. It then
runs the build command, `scons` inside that work directory. The build
products are put onto your host machine in the `sjm` directory. As
soon as the build is finished, the docker container stops and is
removed (`--rm`).

# Preparing the Caltech 101 data

1. Download the [Caltech 101 Dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech101/).
2. Extract the files: `tar -xvzf 101_ObjectCategories.tar.gz`. This should give you a directory called
`101_ObjectCategories` with 102 sub-directories, one for each of the 101 object categories and one background
category.
3. Resize the images to have a maximum width or height of 300 pixels, with preserved aspect ratio. To do this,
I use ImageMagick's `mogrify` command: `mogrify -verbose -resize 300x300 101_ObjectCategories/*/*.jpg`.


# Extracting the SIFT features

On your local host machine, create a directory that the extracted
features will end up in. In the following, I call that
`path_for_extracted_features`.

Use the docker container to run this command:

    docker run --rm -v `pwd`:/work -v [path_to_your_101_Categories_directory]:/images \
      -v [path_for_extracted_features]:/features -w /work/naive_bayes_nearest_neighbor/experiment_1 \
      -e PYTHONPATH=/work sanchom/phd-environment \
      python extract_caltech.py --dataset_path=/images --process_limit=4 --sift_normalization=2.0 \
      --sift_discard_unnormalized --sift_grid_type FIXED_3X3 --sift_first_level_smoothing 0.66 --sift_fast \
      --sift_multiscale --features_directory=/features

This will extract multi-scale SIFT at 4 scales, with a small amount of additional smoothing applied at the first level.
Features are extracted at each level on a 3x3 pixel grid. The first level features are 16x16, and increase by a factor
of 1.5 at each level. This also discards features that from low contrast regions
(`--sift_normalization_threshold 2.0 --sift_discard_unnormalized`).

Now, you should have a directory structure on your local machine at
`path_for_extracted_features` that mirrors that at
`path_to_your_101_Categories`, but with `.sift` files instead of
`.jpeg` files.

# Standard NBNN

Create a `101_categories.txt` file that lists all the 101 object
categories (not BACKGROUND_Google). We ignore the background class as
[suggested by the dataset
creators](http://authors.library.caltech.edu/7694/1/CNS-TR-2007-001.pdf).

Run this:

    docker run --rm -v `pwd`:/work -v [path_to_extracted_features]:/features \
      -w /work sanchom/phd-environment ./naive_bayes_nearest_neighbor/experiment_1/experiment_1 \
      --category_list 101_categories.txt --features_directory /features \
      --alpha [alpha] --trees [trees] --checks [checks] \
      --results_file [results_file] --logtostderr

In our experiments, we fixed alpha=1.6, trees=4, and varied the checks variable depending on the particular experiment
we were performing, but for optimal performance, checks should be greater than 128 (see Figure 4 from our paper).

The NBNN algorithm is implemented in
[`NbnnClassifier::Classify`](https://github.com/sanchom/sjm/blob/master/naive_bayes_nearest_neighbor/nbnn_classifier-inl.h#L92)

# Local NBNN

Create a `101_categories.txt` file that lists all the 101 object
categories (not BACKGROUND_Google). We ignore the background class as
[suggested by the dataset
creators](http://authors.library.caltech.edu/7694/1/CNS-TR-2007-001.pdf).

Run this:

    docker run --rm -v `pwd`:`pwd` -v [path_to_extracted_features]:/features \
    -w `pwd` sanchom/phd-environment ./naive_bayes_nearest_neighbor/experiment_3/experiment_3 \
    --category_list 101_categories.txt --features_directory /features \
    --k [k] --alpha [alpha] --trees [trees] --checks [checks] \
    --results_file [results_file] --logtostderr

In our experiments, we fixed alpha=1.6, trees=4, and varied k and checks depending on the experiment.
For optimal results, checks should be above 1024 (see Figure 4 from our paper), and k should be around 10-20
(see Figure 3 from our paper).

The Local NBNN algorithm is implemented in
[`MergedClassifier::Classify`](https://github.com/sanchom/sjm/blob/master/naive_bayes_nearest_neighbor/merged_classifier.h#L166)

# Spatially Local Coding

This section is being rewritten, but if you're curious, look in the
raw text of this
[README](https://raw.githubusercontent.com/sanchom/sjm/master/README.markdown)
file for a section that's been commented out.

<!--
# Spatially Local Coding

This section explains how to run the experiments described in
_Spatially Local Coding for Object Recognition_.

### Localized Soft Assignment Spatial Pyramid Match

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

### Standard Spatial Pyramid Match

    rm -f /tmp/*.svm; rm -rf [featuredir]; \
    python baseline_experiment.py --dataset_path=[caltechdir] --work_directory=[resultdir] --clobber \
    --sift_normalization_threshold=2.0 --sift_discard_unnormalized \
    --sift_grid_type=FIXED_8X8 --sift_first_level_smoothing=0.66 --features_directory=[featuredir] \
    --process_limit=[num_processes] --num_train=[num_train] --num_test=15 --codeword_locality=1 --pooling=AVERAGE_POOLING \
    --dictionary_training_size=1000000 --clobber_dictionary --pyramid_levels=3 --kmeans_accuracy=0.9 \
    --dictionary=[dictionary_size]:0 --kernel=[intersection|linear]

What makes this Standard SPM is: `--codeword_locality=1 --pooling=AVERAGE_POOLING --pyramid_levels=3`.

### Spatially Local Coding

    rm -f /tmp/*.svm; rm -rf [featuredir]; \
    python baseline_experiment.py --dataset_path=[caltechdir] --work_directory=[resultdir] --clobber \
    --sift_normalization_threshold=2.0 --sift_discard_unnormalized \
    --sift_grid_type=FIXED_8X8 --sift_first_level_smoothing=0.66 --features_directory=[featuredir] \
    --process_limit=[num_processes] --num_train=[num_train] --num_test=15 --codeword_locality=10 --pooling=MAX_POOLING \
    --dictionary_training_size=1000000 --clobber_dictionary --pyramid_levels=1 --kmeans_accuracy=0.9 \
    --dictionary=[dictionary_size]:0 --dictionary=[dictionary_size]:0.75 \
    --dictionary=[dictionary_size]:1.5 --dictionary=[dictionary_size]:3.00 --kernel=linear

What makes this Spatially Local Coding is: `--codeword_locality=10`,
`--pooling=MAX_POOLING`, `--pyramid_levels=1`, and
`--dictionary=[dictionary_size]:0`,
`--dictionary=[dictionary_size]:0.75`,
`--dictionary=[dictionary_size]:1.5`,
`--dictionary=[dictionary_size]:3.00`. The values after
[dictionary_size] are the location weightings to use for each
dictionary.

### Changing the extraction settings

If you are using Caltech 256, pass the `--caltech256` flag.

You can use other `--sift_grid_type`s to get different extraction
densities.  FIXED_8X8 without the `--sift_multiscale` flag extracts
singlescale SIFT every 8 pixels. If you use SCALED_BIN_WIDTH without
the `--sift_multiscale` flag, you'll get singlescale SIFT every 4
pixels. If you use SCALED_DOUBLE_BIN_WIDTH _with_ `--sift_multiscale`,
you'll get 3 scales of SIFT, with the lowest scale being every 8
pixels. If you use SCALED_BIN_WIDTH with `--sift_multiscale`, you'll
get 3 scales of SIFT with the lowest scale being every 4 pixels.

### Inspecting the coding algorithms

There isn't a single place to point you to if you'd like to inspect
the algorithm. Spatially Local Coding is simply building multiple
codebooks, each taking location into account to a different degree.
Then instead of building a spatial pyramid, just build a bag-of-words
histogram for each of those codebooks and concatenate them. That is
the model for an image.

The approximate k-means clustering is implemented at:
[`CodebookBuilder::ClusterApproximately`](https://github.com/sanchom/sjm/blob/master/codebooks/codebook_builder.cc#L170)

Spatial pyramid construction, including the option for coding across
more than a single dictionary is implemented at:
[`SpatialPyramidBuilder::BuildPyramid`](https://github.com/sanchom/sjm/blob/master/spatial_pyramid/spatial_pyramid_builder.cc#L132)

The SPM kernels can be inspected at
[`spatial_pyramid_kernel.cc`](https://github.com/sanchom/sjm/blob/master/spatial_pyramid/spatial_pyramid_kernel.cc).

Our trainer wrapper for extensive cross-validation and one-vs-all SVM
training is at
[`trainer_cli.cc`](https://github.com/sanchom/sjm/blob/master/spatial_pyramid/trainer_cli.cc).
-->