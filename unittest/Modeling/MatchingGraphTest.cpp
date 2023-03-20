#include "gtest/gtest.h"
#include "marco/Modeling/Matching.h"

#include "MatchingCommon.h"
#include "MatchingTestCases.h"

using namespace ::marco::modeling;
using namespace ::marco::modeling::internal;
using namespace ::marco::modeling::matching::test;

TEST(MatchingGraph, singleVariableInsertion)
{
  mlir::MLIRContext context;
  MatchingGraph<Variable, Equation> graph(&context);
  graph.addVariable(Variable("x"));
  EXPECT_TRUE(graph.hasVariable("x"));
}

TEST(MatchingGraph, multipleVariablesInsertion)
{
  mlir::MLIRContext context;
  MatchingGraph<Variable, Equation> graph(&context);

  graph.addVariable(Variable("x"));
  graph.addVariable(Variable("y"));

  EXPECT_TRUE(graph.hasVariable("x"));
  EXPECT_TRUE(graph.hasVariable("y"));
  EXPECT_FALSE(graph.hasVariable("z"));
}

TEST(MatchingGraph, addEquation)
{
  mlir::MLIRContext context;
  MatchingGraph<Variable, Equation> graph(&context);

  Variable x("x", {2});

  Equation eq1("eq1");
  eq1.addIterationRange(Range(0, 1));
  eq1.addVariableAccess(marco::modeling::matching::Access(x, DimensionAccess::constant(0)));
  eq1.addVariableAccess(marco::modeling::matching::Access(x, DimensionAccess::constant(1)));

  graph.addVariable(x);
  graph.addEquation(eq1);

  ASSERT_TRUE(graph.hasEquation("eq1"));
  ASSERT_TRUE(graph.hasEdge("eq1", "x"));
}

TEST(MatchingGraph, testCase1_equationsVariablesCount)
{
  mlir::MLIRContext context;
  auto graph = testCase1(&context);

  size_t amount = 3;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}

TEST(MatchingGraph, testCase2_equationsVariablesCount)
{
  mlir::MLIRContext context;
  auto graph = testCase2(&context);

  size_t amount = 4;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}

TEST(MatchingGraph, testCase3_equationsVariablesCount)
{
  mlir::MLIRContext context;
  auto graph = testCase3(&context);

  size_t amount = 19;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}

TEST(MatchingGraph, testCase4_equationsVariablesCount)
{
  mlir::MLIRContext context;
  auto graph = testCase4(&context);

  size_t amount = 13;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}

TEST(MatchingGraph, testCase5_equationsVariablesCount)
{
  mlir::MLIRContext context;
  auto graph = testCase5(&context);

  size_t amount = 14;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}

TEST(MatchingGraph, testCase6_equationsVariablesCount)
{
  mlir::MLIRContext context;
  auto graph = testCase6(&context);

  size_t amount = 9;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}

TEST(MatchingGraph, testCase7_equationsVariablesCount)
{
  mlir::MLIRContext context;
  auto graph = testCase7(&context);

  size_t amount = 4;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}

TEST(MatchingGraph, testCase8_equationsVariablesCount)
{
  mlir::MLIRContext context;
  auto graph = testCase8(&context);

  size_t amount = 12;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}

TEST(MatchingGraph, testCase9_equationsVariablesCount)
{
  mlir::MLIRContext context;
  auto graph = testCase9(&context);

  size_t amount = 10;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}

TEST(MatchingGraph, testCase10_equationsVariablesCount)
{
  mlir::MLIRContext context;
  auto graph = testCase10(&context);

  size_t amount = 4;
  EXPECT_EQ(graph.getNumberOfScalarEquations(), amount);
  EXPECT_EQ(graph.getNumberOfScalarVariables(), amount);
}
