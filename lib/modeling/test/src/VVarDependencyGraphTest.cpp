#include <gtest/gtest.h>
#include <marco/modeling/VVarDependencyGraph.h>

#include "SCCTestCases.h"

using namespace ::marco::modeling::scc::test;

TEST(VVarDependencyGraph, test1)
{
  std::cout << "TEST CASE 1\n";
  auto graph = testCase1();
  graph.findSCCs();
}

TEST(VVarDependencyGraph, test2)
{
  std::cout << "TEST CASE 2\n";
  auto graph = testCase2();
  graph.findSCCs();
}
