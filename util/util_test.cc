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

// This file tests my C++ utility functions.

// File under test.
#include "util/util.h"

#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

using std::map;
using std::string;
using std::vector;

TEST(UtilTest, SplitStringUsing) {
  vector<string> results;
  results.push_back("garbage");
  sjm::util::SplitStringUsing("Split this string.", " ", &results);
  ASSERT_EQ(results[0], "Split");
  ASSERT_EQ(results[1], "this");
  ASSERT_EQ(results[2], "string.");
}

TEST(UtilTest, HasKey) {
  map<string, int> test_map;
  test_map["a"] = 1;
  ASSERT_TRUE(sjm::util::HasKey(test_map, "a"));
  ASSERT_FALSE(sjm::util::HasKey(test_map, "b"));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
