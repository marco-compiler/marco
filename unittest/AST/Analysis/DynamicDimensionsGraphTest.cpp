#include "marco/AST/Analysis/DynamicDimensionsGraph.h"
#include "gtest/gtest.h"

using namespace ::marco;
using namespace ::marco::ast;

TEST(DynamicDimensionsGraph, emptyGraph)
{
  DynamicDimensionsGraph graph;

  EXPECT_EQ(graph.getNumOfNodes(), 1);
  EXPECT_FALSE(graph.hasCycles());
}

TEST(DynamicDimensionsGraph, numberOfNodes)
{
  SourceRange loc = SourceRange::unknown();

  llvm::SmallVector<const Member*, 2> inputMembers;
  llvm::SmallVector<const Member*, 1> outputMembers;
  llvm::SmallVector<const Member*, 1> protectedMembers;

  auto x = Member::build(
      loc, "x",
      makeType<BuiltInType::Real>(),
      TypePrefix(ParameterQualifier::none, IOQualifier::input),
      true);

  inputMembers.push_back(x.get());

  auto y = Member::build(
      loc, "y",
      makeType<BuiltInType::Real>(),
      TypePrefix(ParameterQualifier::none, IOQualifier::input),
      true);

  inputMembers.push_back(y.get());

  auto z = Member::build(
      loc, "z",
      makeType<BuiltInType::Real>(),
      TypePrefix(ParameterQualifier::none, IOQualifier::output),
      true);

  outputMembers.push_back(z.get());

  auto t = Member::build(
      loc, "t",
      makeType<BuiltInType::Real>(),
      TypePrefix(ParameterQualifier::none, IOQualifier::none),
      false);

  protectedMembers.push_back(t.get());

  DynamicDimensionsGraph graph;

  graph.addMembersGroup(inputMembers, true);
  graph.addMembersGroup(outputMembers, true);
  graph.addMembersGroup(protectedMembers, true);

  EXPECT_EQ(graph.getNumOfNodes(), 5);
}

TEST(DynamicDimensionsGraph, intraGroupDependencies)
{
  SourceRange loc = SourceRange::unknown();
  llvm::SmallVector<const Member*, 2> members;

  auto x = Member::build(
      loc, "x",
      makeType<BuiltInType::Real>(),
      TypePrefix(ParameterQualifier::none, IOQualifier::input),
      true);

  members.push_back(x.get());

  auto y = Member::build(
      loc, "y",
      makeType<BuiltInType::Real>(),
      TypePrefix(ParameterQualifier::none, IOQualifier::input),
      true);

  members.push_back(y.get());

  DynamicDimensionsGraph graph;

  graph.addMembersGroup(members, true);
  graph.discoverDependencies();

  auto postOrderVisit = graph.postOrder();

  EXPECT_EQ(postOrderVisit.size(), 2);
  EXPECT_EQ(postOrderVisit[0]->getName(), "x");
  EXPECT_EQ(postOrderVisit[1]->getName(), "y");
}

TEST(DynamicDimensionsGraph, interGroupDependencies)
{
  SourceRange loc = SourceRange::unknown();

  llvm::SmallVector<const Member*, 1> inputMembers;
  llvm::SmallVector<const Member*, 1> outputMembers;

  auto x = Member::build(
      loc, "x",
      makeType<BuiltInType::Real>(),
      TypePrefix(ParameterQualifier::none, IOQualifier::input),
      true);

  inputMembers.push_back(x.get());

  auto expression = Expression::call(
      loc, makeType<BuiltInType::Integer>(),
      Expression::reference(loc, makeType<BuiltInType::Integer>(), "size"),
      llvm::makeArrayRef({
          Expression::reference(loc, makeType<BuiltInType::Integer>(), "x"),
          Expression::constant(loc, makeType<BuiltInType::Integer>(), 1)
      }));

  auto y = Member::build(
      loc, "y",
      makeType<BuiltInType::Real>(std::move(expression)),
      TypePrefix(ParameterQualifier::none, IOQualifier::output),
      true);

  outputMembers.push_back(y.get());

  DynamicDimensionsGraph graph;

  graph.addMembersGroup(inputMembers, true);
  graph.addMembersGroup(outputMembers, true);
  graph.discoverDependencies();

  auto postOrderVisit = graph.postOrder();

  EXPECT_EQ(postOrderVisit.size(), 2);
  EXPECT_EQ(postOrderVisit[0]->getName(), "x");
  EXPECT_EQ(postOrderVisit[1]->getName(), "y");
}
