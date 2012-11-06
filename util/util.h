// Copyright (c) 2011-2012, Sancho McCann

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

#ifndef UTIL_UTIL_H_
#define UTIL_UTIL_H_

#include <cstdlib>
#include <set>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "boost/thread.hpp"

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

template<typename T>
inline bool HasKey(const T& keyed_collection, const typename T::key_type& key) {
  return keyed_collection.find(key) != keyed_collection.end();
}

inline void PollForAvailablePoolSpace(
    int thread_limit, int ms_wait, std::vector<boost::thread*>* thread_pool) {
  while (static_cast<int>(thread_pool->size()) >= thread_limit) {
    for (std::vector<boost::thread*>::iterator thread_it = thread_pool->begin();
         thread_it != thread_pool->end();
         ++thread_it) {
      if ((*thread_it)->timed_join(boost::posix_time::milliseconds(ms_wait))) {
        delete (*thread_it);
        thread_pool->erase(thread_it);
        break;
      }
    }
  }
}

inline void JoinWithPool(std::vector<boost::thread*>* thread_pool) {
  for (std::vector<boost::thread*>::iterator thread_it = thread_pool->begin();
       thread_it != thread_pool->end();
       ++thread_it) {
    (*thread_it)->join();
    delete (*thread_it);
  }
  thread_pool->clear();
}

}}  // End namespaces sjm, util

#endif  // UTIL_UTIL_H_
