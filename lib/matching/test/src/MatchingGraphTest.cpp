#include <gtest/gtest.h>
#include <marco/matching/Matching.h>

#include "Common.h"
#include "TestCases.h"

using namespace marco::matching;
using namespace marco::matching::detail;

TEST(Matching, singleVariableInsertion)
{
  MatchingGraph<Variable, Equation> graph;
  graph.addVariable(Variable("x"));
  EXPECT_TRUE(graph.hasVariable("x"));
}

TEST(Matching, multipleVariablesInsertion)
{
  MatchingGraph<Variable, Equation> graph;

  graph.addVariable(Variable("x"));
  graph.addVariable(Variable("y"));

  EXPECT_TRUE(graph.hasVariable("x"));
  EXPECT_TRUE(graph.hasVariable("y"));
  EXPECT_FALSE(graph.hasVariable("z"));
}

TEST(Matching, multidimensionalVariableInsertion)
{
  MatchingGraph<Variable, Equation> graph;
  graph.addVariable(Variable("x", { 2, 3, 4 }));

  auto var = graph.getVariable("x");
  EXPECT_EQ(var.getRank(), 3);
  EXPECT_EQ(var.getDimensionSize(0), 2);
  EXPECT_EQ(var.getDimensionSize(1), 3);
  EXPECT_EQ(var.getDimensionSize(2), 4);
}

TEST(Matching, addEquation)
{
  MatchingGraph<Variable, Equation> graph;

  Variable x("x", { 2 });

  Equation eq1("eq1");
  eq1.addIterationRange(Range(0, 1));
  eq1.addVariableAccess(Access(x, DimensionAccess::constant(0)));
  eq1.addVariableAccess(Access(x, DimensionAccess::constant(1)));

  graph.addVariable(x);
  graph.addEquation(eq1);

  ASSERT_TRUE(graph.hasEquation("eq1"));
  ASSERT_TRUE(graph.hasEdge("eq1", "x"));
}

TEST(Matching, testCase1_EquationsVariablesCount)
{
  auto graph = testCase1();

  size_t amount = 3;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}

TEST(Matching, testCase2_EquationsVariablesCount)
{
  auto graph = testCase2();

  size_t amount = 4;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}

TEST(Matching, testCase3_EquationsVariablesCount)
{
  auto graph = testCase3();

  size_t amount = 19;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}

TEST(Matching, testCase4_EquationsVariablesCount)
{
  auto graph = testCase4();

  size_t amount = 13;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}

TEST(Matching, testCase5_EquationsVariablesCount)
{
  auto graph = testCase5();

  size_t amount = 14;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}

TEST(Matching, testCase6_EquationsVariablesCount)
{
  auto graph = testCase6();

  size_t amount = 9;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}

TEST(Matching, testCase7_EquationsVariablesCount)
{
  auto graph = testCase7();

  size_t amount = 4;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}

TEST(Matching, testCase8_EquationsVariablesCount)
{
  auto graph = testCase8();

  size_t amount = 12;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}

TEST(Matching, testCase9_EquationsVariablesCount)
{
  auto graph = testCase9();

  size_t amount = 10;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}

TEST(Matching, testCase10_EquationsVariablesCount)
{
  auto graph = testCase10();

  size_t amount = 4;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}
