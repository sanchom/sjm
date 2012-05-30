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

// These utilities assist saving and reading of sift descriptor
// protocol buffers with accessory information.

#include <string>
#include <tr1/cstdint>

namespace sjm {
namespace sift {
class SiftDescriptor;
class DescriptorSet;
class ExtractionParameters;
}} // namespace

namespace sjm {
namespace sift {
// Writes a descriptor set to file in two parts: parameters, then the
// entire descriptor set, including the parameters again.
// The format of the resulting binary file is like this:
// <size of parameter data>
// <parameter data>
// <size of entire message>
// <entire message>
void WriteDescriptorSetToFile(const sjm::sift::DescriptorSet &descriptors,
                              const std::string &filename);

// Reads only the parameters from a file. This clears any non-default
// content already in parameters.
void ReadParametersFromFile(const std::string &filename,
                            sjm::sift::ExtractionParameters *parameters);

// Reads the descriptor set, including the parameters. This clears any
// non-default content already in descriptors.
void ReadDescriptorSetFromFile(const std::string &filename,
                               sjm::sift::DescriptorSet *descriptors);

// Converts an instance of SiftDescriptor to a uint8_t array.  This
// throws out the location information unless it's added using the
// alpha weighting as extra dimensions at the end. Returns the number
// of uint8_ts written to the destination array.
int ConvertProtobufDescriptorToWeightedArray(
    const sjm::sift::SiftDescriptor &descriptor,
    const float alpha,
    uint8_t *destination);

} // namespace sift
} // namespace sjm
