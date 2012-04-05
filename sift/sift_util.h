// Copyright 2011 Sancho McCann
// Author: Sancho McCann
//
// These utilities assist saving and reading of sift descriptor
// protocol buffers.

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
