DEPENDENCIES
============
- Python and NumPy
- Boost C++ Libraries [http://www.boost.org]
- OpenCV (and libpng, libjpeg, libjasper, libtiff) [http://opencv.willowgarage.com/wiki/]
- FLANN [http://www.cs.ubc.ca/~mariusm/index.php/FLANN/FLANN]
- VlFeat [http://www.vlfeat.org/]
- Google libraries
  - Google Test (gtest) [http://code.google.com/p/googletest/]
  - Google Command Line Flags (gflags) [http://code.google.com/p/gflags/]
  - Google Logging (glog) [http://code.google.com/p/google-glog/]
  - Google Protocol Buffers (protoc) [http://code.google.com/p/protobuf/]
- SCons build system [http://www.scons.org/]


BUILDING
========

First, install all of the dependencies listed above and note where their headers and libraries are installed to.

Then, clone this repository to your local machine: `git clone git@github.com:sanchom/sjm_ubc.git`

Add the root directory of the repository to your `$PYTHONPATH` environment variable.
Something like this: `export PYTHONPATH=~/sjm_ubc:$PYTHONPATH`. You can put that in your `.bash_profile`.

You'll also need to update the environment variables `$LIBRARY_PATH` and `$CPLUS_INCLUDE_PATH` to contain the
directories where the dependencies were installed to. The Scons build script will use these environment variables
to search for the dependencies.

You can do this in your `.bash_profile`:

    export LIBRARY_PATH=/custom/lib:$LIBRARY_PATH;
    export CPLUS_INCLUDE_PATH=/custom/include:$CPLUS_INCLUDE_PATH;

Or, you can do this at the command line before running `scons`:

    LIBRARY_PATH=/custom/lib:$LIBRARY_PATH; CPLUS_INCLUDE_PATH=/custom/include:$CPLUS_INCLUDE_PATH; scons;

In either case, just run `scons` in the root directory (where the `SConstruct` file is located).

To install the resultant binaries in a stand-alone location, run `scons install BIN_PREFIX=[directory]`
to install them in `directory`. You should either install them in a directory that's already in your `$PATH`
environment variable (like `/usr/local/bin`, probably) or add the directory you chose to your `$PATH` environment
variable. This is necessary because some of the Python scripts that run the experiments will call these
tools via the subprocess module, which needs to be able to find them via the `$PATH` environment variable.

USING
=====

NBNN and Local NBNN Experiments
-------------------------------

Spatial Pyramid Experiments
---------------------------