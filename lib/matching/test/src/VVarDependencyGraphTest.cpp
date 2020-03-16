#include "gtest/gtest.h"

#include "modelica/matching/SVarDependencyGraph.hpp"
#include "modelica/matching/VVarDependencyGraph.hpp"

using namespace std;
using namespace llvm;
using namespace modelica;

TEST(VVarDependencyGraphTest, countTest)
{
	EntryModel model;
	model.emplaceVar(
			"leftVar",
			ModExp(ModConst(0, 1, 2, 3), ModType(BultinModTypes::INT, 2, 2)));
	model.emplaceEquation(
			ModExp::at(
					ModExp("leftVar", ModType(BultinModTypes::INT, 2, 2)),
					ModExp::induction(ModConst(0))),
			ModConst(3),
			{ InductionVar(1, 3) });

	VVarDependencyGraph graph(model);
	EXPECT_EQ(graph.count(), 1);
}

static auto makeModel()
{
	EntryModel model;
	model.emplaceVar(
			"leftVar", ModExp(ModConst(0, 1), ModType(BultinModTypes::INT, 2)));
	model.emplaceVar(
			"rightVar", ModExp(ModConst(0, 1), ModType(BultinModTypes::INT, 2)));
	model.emplaceEquation(
			ModExp::at(
					ModExp("leftVar", ModType(BultinModTypes::INT, 2, 2)),
					ModExp::induction(ModConst(0))),
			ModConst(3),
			{ InductionVar(0, 2) });

	model.emplaceEquation(
			ModExp::at(
					ModExp("rightVar", ModType(BultinModTypes::INT, 2, 2)),
					ModExp::induction(ModConst(0))),
			ModExp::at(
					ModExp("leftVar", ModType(BultinModTypes::INT, 2, 2)),
					ModExp::induction(ModConst(0))),
			{ InductionVar(0, 2) });
	return model;
}

TEST(VVarDependencyGraphTest, paritionTest)
{
	auto model = makeModel();
	VVarDependencyGraph graph(model);
	auto scc = graph.getSCC();
	EXPECT_EQ(scc.count(), 2);
}

TEST(VVarDependencyGraphTest, graphIteratorTest)
{
	auto model = makeModel();
	VVarDependencyGraph graph(model);
	auto sccContent = graph.getSCC();

	int visitedNodes = 0;
	for (auto& scc : sccContent)
		for (auto& vertex : scc.range(graph))
			visitedNodes++;

	EXPECT_EQ(visitedNodes, graph.count());
}

TEST(VVarDependencyGraphTest, scalarGraph)
{
	auto model = makeModel();
	VVarDependencyGraph graph(model);
	auto sccContent = graph.getSCC();

	int visitedNodes = 0;
	for (const auto& scc : sccContent)
	{
		SVarDepencyGraph scalarGraph(graph, scc);
		visitedNodes += scalarGraph.count();
	}
	EXPECT_EQ(visitedNodes, 4);
}
