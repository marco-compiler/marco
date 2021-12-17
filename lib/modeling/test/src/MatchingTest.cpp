#include <gtest/gtest.h>
#include <marco/modeling/Matching.h>

#include "MatchingTestCases.h"

using namespace marco::modeling::matching::test;

TEST(Matching, testCase1)
{
  auto graph = testCase1();
  ASSERT_TRUE(graph.match());
}

TEST(Matching, testCase2)
{
  auto graph = testCase2();
  ASSERT_TRUE(graph.match());
}

TEST(Matching, testCase3)
{
  auto graph = testCase3();
  ASSERT_TRUE(graph.match());
}

TEST(Matching, testCase4)
{
  auto graph = testCase4();
  ASSERT_TRUE(graph.match());
}

TEST(Matching, testCase5)
{
  auto graph = testCase5();
  ASSERT_TRUE(graph.match());
}

TEST(Matching, testCase6)
{
  auto graph = testCase6();
  ASSERT_TRUE(graph.match());
}

TEST(Matching, testCase7)
{
  auto graph = testCase7();
  ASSERT_TRUE(graph.match());
}

TEST(Matching, testCase8)
{
  auto graph = testCase8();
  ASSERT_TRUE(graph.match());
}

TEST(Matching, testCase9)
{
  auto graph = testCase9();
  ASSERT_TRUE(graph.match());
}

TEST(Matching, testCase10)
{
  auto graph = testCase10();
  ASSERT_TRUE(graph.match());
}
