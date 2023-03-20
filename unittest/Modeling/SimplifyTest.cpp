#include "gtest/gtest.h"
#include "marco/Modeling/Matching.h"

#include "MatchingTestCases.h"

using namespace ::marco::modeling::matching::test;

TEST(Simplify, testCase1)
{
  mlir::MLIRContext context;
  auto graph = testCase1(&context);
  ASSERT_TRUE(graph.simplify());
}

TEST(Simplify, testCase2)
{
  mlir::MLIRContext context;
  auto graph = testCase2(&context);
  ASSERT_TRUE(graph.simplify());
}

TEST(Simplify, testCase3)
{
  mlir::MLIRContext context;
  auto graph = testCase3(&context);
  ASSERT_TRUE(graph.simplify());
}

TEST(Simplify, testCase4)
{
  mlir::MLIRContext context;
  auto graph = testCase4(&context);
  ASSERT_TRUE(graph.simplify());
}

TEST(Simplify, testCase5)
{
  mlir::MLIRContext context;
  auto graph = testCase5(&context);
  ASSERT_TRUE(graph.simplify());
}

TEST(Simplify, testCase6)
{
  mlir::MLIRContext context;
  auto graph = testCase6(&context);
  ASSERT_TRUE(graph.simplify());
}

TEST(Simplify, testCase7)
{
  mlir::MLIRContext context;
  auto graph = testCase7(&context);
  ASSERT_TRUE(graph.simplify());
}

TEST(Simplify, testCase8)
{
  mlir::MLIRContext context;
  auto graph = testCase8(&context);
  ASSERT_TRUE(graph.simplify());
}

TEST(Simplify, testCase9)
{
  mlir::MLIRContext context;
  auto graph = testCase9(&context);
  ASSERT_TRUE(graph.simplify());
}

TEST(Simplify, testCase10)
{
  mlir::MLIRContext context;
  auto graph = testCase10(&context);
  ASSERT_TRUE(graph.simplify());
}
