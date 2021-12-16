#include <gtest/gtest.h>
#include <marco/matching/Matching.h>

#include "MatchingTestCases.h"

using namespace marco::matching;
using namespace marco::matching::detail;

TEST(Simplify, testCase1)
{
  auto graph = testCase1();
  ASSERT_TRUE(graph.simplify());
}

TEST(Simplify, testCase2)
{
  auto graph = testCase2();
  ASSERT_TRUE(graph.simplify());
}

TEST(Simplify, testCase3)
{
  auto graph = testCase3();
  ASSERT_TRUE(graph.simplify());
}

TEST(Simplify, testCase4)
{
  auto graph = testCase4();
  ASSERT_TRUE(graph.simplify());
}

TEST(Simplify, testCase5)
{
  auto graph = testCase5();
  ASSERT_TRUE(graph.simplify());
}

TEST(Simplify, testCase6)
{
  auto graph = testCase6();
  ASSERT_TRUE(graph.simplify());
}

TEST(Simplify, testCase7)
{
  auto graph = testCase7();
  ASSERT_TRUE(graph.simplify());
}

TEST(Simplify, testCase8)
{
  auto graph = testCase8();
  ASSERT_TRUE(graph.simplify());
}

TEST(Simplify, testCase9)
{
  auto graph = testCase9();
  ASSERT_TRUE(graph.simplify());
}

TEST(Simplify, testCase10)
{
  auto graph = testCase10();
  ASSERT_TRUE(graph.simplify());
}
