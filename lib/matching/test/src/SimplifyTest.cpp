#include <gtest/gtest.h>
#include <marco/matching/Matching.h>

#include "TestCases.h"

using namespace marco::matching;
using namespace marco::matching::detail;

TEST(Matching, testCase1_simplify)
{
  auto graph = testCase1();
  ASSERT_TRUE(graph.simplify());
}

TEST(Matching, testCase2_simplify)
{
  auto graph = testCase2();
  ASSERT_TRUE(graph.simplify());
}

TEST(Matching, testCase3_simplify)
{
  auto graph = testCase3();
  ASSERT_TRUE(graph.simplify());
}

TEST(Matching, testCase4_simplify)
{
  auto graph = testCase4();
  ASSERT_TRUE(graph.simplify());
}

TEST(Matching, testCase5_simplify)
{
  auto graph = testCase5();
  ASSERT_TRUE(graph.simplify());
}

TEST(Matching, testCase6_simplify)
{
  auto graph = testCase6();
  ASSERT_TRUE(graph.simplify());
}

TEST(Matching, testCase7_simplify)
{
  auto graph = testCase7();
  ASSERT_TRUE(graph.simplify());

  //auto eq1 = graph.getEquationVertex("eq1");
  graph.dump();
}

TEST(Matching, testCase8_simplify)
{
  auto graph = testCase8();
  ASSERT_TRUE(graph.simplify());
}

TEST(Matching, testCase9_simplify)
{
  auto graph = testCase9();
  ASSERT_TRUE(graph.simplify());
}

TEST(Matching, testCase10_simplify)
{
  auto graph = testCase10();
  ASSERT_TRUE(graph.simplify());
}
