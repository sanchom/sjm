// Copyright 2011 Sancho McCann
// Author: Sancho McCann

// This file tests my C++ utility functions.

// File under test.
#include "util/util.h"

#include <string>
#include <vector>

#include "gtest/gtest.h"

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

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
