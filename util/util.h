// Copyright 2011 Sancho McCann
// Author: Sancho McCann

#ifndef UTIL_UTIL_H_
#define UTIL_UTIL_H_

#include <cstdlib>
#include <set>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"

#include "glog/logging.h"

namespace sjm {
namespace util {

inline void SplitStringUsing(
    const std::string& input, const std::string& separators,
    std::vector<std::string>* results) {
  results->clear();
  boost::split(*results, input, boost::is_any_of(separators));
}

// Expands a leading '~' in a path into a full path by replacing the
// '~' with the current user's complete path to their home
// folder. Does nothing to paths that don't have a leading '~'.
inline std::string expand_user(std::string path) {
  if (!path.empty() and path[0] == '~') {
    CHECK(path.size() == 1 || path[1] == '/') <<
        "Attempting to expand malformed path.";
    const char* home = getenv("HOME");
    if (home || (home = getenv("USERPROFILE"))) {
      // Replace the first character with the home directory.
      path.replace(0, 1, home);
    }
  }
  return path;
}

inline
void ReadFileToStringOrDie(const std::string& filename, std::string* dest) {
  dest->clear();
  FILE* f = fopen(sjm::util::expand_user(filename).c_str(), "rb");
  CHECK(f != NULL) << "Error opening " << filename;
  const size_t kBufferSize = 1024;
  char buffer[kBufferSize];
  size_t num_read = 0;
  while ((num_read = fread(buffer, sizeof(char), kBufferSize, f)) > 0) {
    dest->append(buffer, num_read);
  }
  CHECK_EQ(0, fclose(f));
}

inline
void ReadLinesFromFileIntoVectorOrDie(const std::string& filename,
                                      std::vector<std::string>* dest) {
  dest->clear();
  std::string data;
  ReadFileToStringOrDie(sjm::util::expand_user(filename), &data);
  std::vector<std::string> file_lines;
  boost::split(file_lines, data, boost::is_any_of("\n"));
  for (size_t i = 0; i < file_lines.size(); ++i) {
    if (!file_lines[i].empty()) {
      dest->push_back(file_lines[i]);
    }
  }
}

inline
void WriteStringToFileOrDie(const std::string& filename,
                            const std::string& source) {
  FILE* f = fopen(sjm::util::expand_user(filename).c_str(), "wb");
  CHECK(f != NULL) << "Error opening " << filename << " for writing.";
  CHECK_EQ(source.size(), fwrite(source.data(), 1, source.size(), f));
  CHECK_EQ(0, fclose(f));
}

inline
void AppendStringToFileOrDie(const std::string& filename,
                             const std::string& source) {
  FILE* f = fopen(sjm::util::expand_user(filename).c_str(), "a");
  CHECK(f != NULL) << "Error opening " << filename << " for append.";
  CHECK_EQ(source.size(), fwrite(source.data(), 1, source.size(), f));
  CHECK_EQ(0, fclose(f));
}
}}  // End namespaces sjm, util

#endif  // UTIL_UTIL_H_
