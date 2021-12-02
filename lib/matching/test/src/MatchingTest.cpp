#include <gtest/gtest.h>
#include <marco/matching/Matching.h>

#include "TestCases.h"

using namespace marco::matching;
using namespace marco::matching::detail;

TEST(Matching, testCase1_matching)
{
  auto graph = testCase1();
  ASSERT_TRUE(graph.match());

  graph.dump();
}

TEST(Matching, testCase2_matching)
{
  auto graph = testCase2();
  ASSERT_TRUE(graph.match());

  graph.dump();
}

TEST(Matching, testCase3_matching)
{
  auto graph = testCase3();
  bool result = graph.match();
  graph.dump();
  ASSERT_TRUE(result);
}

TEST(Matching, testCase4_matching)
{
  auto graph = testCase4();
  ASSERT_TRUE(graph.match());

  graph.dump();
}

TEST(Matching, testCase5_matching)
{
  auto graph = testCase5();
  ASSERT_TRUE(graph.match());

  graph.dump();
}

TEST(Matching, testCase6_matching)
{
  auto graph = testCase6();
  ASSERT_TRUE(graph.match());

  graph.dump();
}

TEST(Matching, testCase7_matching)
{
  auto graph = testCase7();
  ASSERT_TRUE(graph.match());

  graph.dump();
}

TEST(Matching, testCase8_matching)
{
  auto graph = testCase8();
  ASSERT_TRUE(graph.match());

  graph.dump();
}

TEST(Matching, testCase9_matching)
{
  auto graph = testCase9();
  bool result = graph.match();
  //graph.dump();
  ASSERT_TRUE(result);
}

TEST(Matching, testCase10_matching)
{
  auto graph = testCase10();
  ASSERT_TRUE(graph.match());

  graph.dump();
}
