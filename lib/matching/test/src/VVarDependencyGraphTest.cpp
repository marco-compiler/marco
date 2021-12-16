#include <gtest/gtest.h>
#include <marco/matching/VVarDependencyGraph.h>

#include "SCCTestCases.h"

using namespace marco::scc;
using namespace marco::scc::detail;

TEST(VVarDependencyGraph, test)
{
  auto graph = testCase1();
  graph.findSCCs();
  std::cout << "Prova";
}