#include <gtest/gtest.h>
#include <marco/modeling/VVarDependencyGraph.h>

#include "SCCTestCases.h"

using namespace marco::modeling::scc::test;

TEST(VVarDependencyGraph, test)
{
  auto graph = testCase1();
  graph.findSCCs();
  std::cout << "Prova";
}