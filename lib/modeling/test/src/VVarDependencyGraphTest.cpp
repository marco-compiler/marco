#include <gtest/gtest.h>
#include <marco/modeling/VVarDependencyGraph.h>

#include "SCCTestCases.h"

using namespace ::marco::modeling::scc::test;

TEST(VVarDependencyGraph, test1)
{
  std::cout << "TEST CASE 1\n";
  auto graph = testCase1();
  graph.getCircularDependencies();
  //auto SCCs = graph.getCircularDependencies();
  //std::cout << "Number of SCCs: " << SCCs.size() << "\n";
}

TEST(VVarDependencyGraph, test2)
{
  std::cout << "TEST CASE 2\n";
  auto graph = testCase2();
  graph.getCircularDependencies();
  //auto SCCs = graph.getCircularDependencies();
  //std::cout << "Number of SCCs: " << SCCs.size() << "\n";
}

TEST(VVarDependencyGraph, test3)
{
  std::cout << "TEST CASE 3\n";
  auto graph = testCase3();
  graph.getCircularDependencies();
  //auto SCCs = graph.getCircularDependencies();
  //std::cout << "Number of SCCs: " << SCCs.size() << "\n";
}
