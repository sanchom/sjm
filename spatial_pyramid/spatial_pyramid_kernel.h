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

// The functions in this file are used to compute various similarity
// measures of Spatial Pyramid representations output from
// SpatialPyramidBuilder::BuildPyramid.

#ifndef SPATIAL_PYRAMID_SPATIAL_PYRAMID_KERNEL_H_
#define SPATIAL_PYRAMID_SPATIAL_PYRAMID_KERNEL_H_

namespace sjm {
namespace spatial_pyramid {

// Forward declaration.
class SpatialPyramid;
class SparseVectorFloat;

// Computes the Spatial Pyramid Match Kernel as described in
// Lazebnik's 2006 CVPR paper. The input pyramids must have the same
// geometry (number of levels, and number of spatial bins per level),
// and the num_levels for which the kernel will be computed can't be
// more than the number of levels available in the pyramids.
//
// Internally, this implementation just computes histogram
// intersections between the different bins in the hierarchical
// histogram. If you want normalization, or max pooling, you need to
// do this before you pass in the
// pyramids. SpatialPyramidBuilder::BuildPyramid does this, so this
// detail doesn't matter unless you implement your own pyramid
// builder.
float SpmKernel(const SpatialPyramid& pyramid_a,
                const SpatialPyramid& pyramid_b,
                const int num_levels);

// Computes a simple linear kernel over the unweighted, concatenated
// histograms (a dot product).
float LinearKernel(const SpatialPyramid& pyramid_a,
                   const SpatialPyramid& pyramid_b);

// Computes the histogram intersection over two sparse vectors.
float HistogramIntersection(const SparseVectorFloat& a,
                            const SparseVectorFloat& b);

// Concatenates together the histograms of the spatial pyramid into a
// single sparse histogram represenetation. If the pyramid has only
// the first level, the returned vector will just equal that
// bag-of-words histogram. If the pyramid has two levels (1x1, and
// 2x2), the bag-of-words histogram bins will be first in the
// concatenation, the next level's histograms are written in row major
// order next. The total dimensionality of the unrolled representation
// is returned in result_dimensions.
void UnrollHistograms(const SpatialPyramid& pyramid,
                      SparseVectorFloat* result_histogram,
                      int* result_dimensions = 0);

// Does a dot product of two sparse vectors.
float Dot(const SparseVectorFloat& a, const SparseVectorFloat& b);
}}  // namespace.

#endif  // SPATIAL_PYRAMID_SPATIAL_PYRAMID_KERNEL_H_
