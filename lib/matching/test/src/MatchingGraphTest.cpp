#include <gtest/gtest.h>
#include <marco/matching/Matching.h>

#include "Common.h"
#include "TestCases.h"

using namespace marco::matching;
using namespace marco::matching::detail;

TEST(MatchingGraph, singleVariableInsertion)
{
  MatchingGraph<Variable, Equation> graph;
  graph.addVariable(Variable("x"));
  EXPECT_TRUE(graph.hasVariable("x"));
}

TEST(MatchingGraph, multipleVariablesInsertion)
{
  MatchingGraph<Variable, Equation> graph;

  graph.addVariable(Variable("x"));
  graph.addVariable(Variable("y"));

  EXPECT_TRUE(graph.hasVariable("x"));
  EXPECT_TRUE(graph.hasVariable("y"));
  EXPECT_FALSE(graph.hasVariable("z"));
}

TEST(MatchingGraph, multidimensionalVariableInsertion)
{
  MatchingGraph<Variable, Equation> graph;
  graph.addVariable(Variable("x", { 2, 3, 4 }));

  auto var = graph.getVariable("x");
  EXPECT_EQ(var.getRank(), 3);
  EXPECT_EQ(var.getDimensionSize(0), 2);
  EXPECT_EQ(var.getDimensionSize(1), 3);
  EXPECT_EQ(var.getDimensionSize(2), 4);
}

TEST(MatchingGraph, addEquation)
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

TEST(MatchingGraph, testCase1_equationsVariablesCount)
{
  auto graph = testCase1();

  size_t amount = 3;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}

TEST(MatchingGraph, testCase2_equationsVariablesCount)
{
  auto graph = testCase2();

  size_t amount = 4;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}

TEST(MatchingGraph, testCase3_equationsVariablesCount)
{
  auto graph = testCase3();

  size_t amount = 19;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}

TEST(MatchingGraph, testCase4_equationsVariablesCount)
{
  auto graph = testCase4();

  size_t amount = 13;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}

TEST(MatchingGraph, testCase5_equationsVariablesCount)
{
  auto graph = testCase5();

  size_t amount = 14;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}

TEST(MatchingGraph, testCase6_equationsVariablesCount)
{
  auto graph = testCase6();

  size_t amount = 9;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}

TEST(MatchingGraph, testCase7_equationsVariablesCount)
{
  auto graph = testCase7();

  size_t amount = 4;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}

TEST(MatchingGraph, testCase8_equationsVariablesCount)
{
  auto graph = testCase8();

  size_t amount = 12;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}

TEST(MatchingGraph, testCase9_equationsVariablesCount)
{
  auto graph = testCase9();

  size_t amount = 10;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}

TEST(MatchingGraph, testCase10_equationsVariablesCount)
{
  auto graph = testCase10();

  size_t amount = 4;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}