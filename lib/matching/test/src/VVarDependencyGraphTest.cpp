#include "gtest/gtest.h"
#include <iterator>

#include "llvm/InitializePasses.h"
#include "modelica/matching/KhanAdjacentAlgorithm.hpp"
#include "modelica/matching/SVarDependencyGraph.hpp"
#include "modelica/matching/Schedule.hpp"
#include "modelica/matching/VVarDependencyGraph.hpp"

using namespace std;
using namespace llvm;
using namespace modelica;

TEST(VVarDependencyGraphTest, countTest)
{
	Model model;
	model.emplaceVar(
			"leftVar",
			ModExp(ModConst(0, 1, 2, 3), ModType(BultinModTypes::INT, 2, 2)));
	model.emplaceEquation(
			ModExp::at(
					ModExp("leftVar", ModType(BultinModTypes::INT, 2, 2)),
					ModExp::induction(ModConst(0))),
			ModConst(3),
			"",
			{ { 1, 3 } });

	VVarDependencyGraph graph(model);
	EXPECT_EQ(graph.count(), 1);
}

static auto makeModel()
{
	Model model;
	model.emplaceVar(
			"leftVar", ModExp(ModConst(0, 1), ModType(BultinModTypes::INT, 2)));
	model.emplaceVar(
			"rightVar", ModExp(ModConst(0, 1), ModType(BultinModTypes::INT, 2)));
	model.emplaceEquation(
			ModExp::at(
					ModExp("leftVar", ModType(BultinModTypes::INT, 2, 2)),
					ModExp::induction(ModConst(0))),
			ModConst(3),
			"",
			{ { 0, 2 } });

	model.emplaceEquation(
			ModExp::at(
					ModExp("rightVar", ModType(BultinModTypes::INT, 2, 2)),
					ModExp::induction(ModConst(0))),
			ModExp::at(
					ModExp("leftVar", ModType(BultinModTypes::INT, 2, 2)),
					ModExp::induction(ModConst(0))),
			"",
			{ { 0, 2 } });
	return model;
}

TEST(VVarDependencyGraphTest, paritionTest)
{
	auto model = makeModel();
	VVarDependencyGraph graph(model);
	SccLookup scc(graph);
	EXPECT_EQ(scc.count(), 2);
}

TEST(VVarDependencyGraphTest, graphIteratorTest)
{
	auto model = makeModel();
	VVarDependencyGraph graph(model);
	SccLookup sccContent(graph);

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
	SccLookup sccContent(graph);

	int visitedNodes = 0;
	for (const auto& scc : sccContent)
	{
		SVarDepencyGraph scalarGraph(graph, scc);
		visitedNodes += scalarGraph.count();
	}
	EXPECT_EQ(visitedNodes, 4);
}

TEST(VVarDependencyGraphTest, testOder)
{
	auto model = makeModel();
	VVarDependencyGraph graph(model);
	SccLookup sccContent(graph);
	SmallVector<SVarDepencyGraph, 0> sccs;
	SmallVector<size_t, 0> execOrder;
	int cutCount = 0;

	for (const auto& scc : sccContent)
	{
		sccs.emplace_back(graph, scc);
		sccs.back().topoOrder(
				[&](size_t order) { execOrder.emplace_back(order); },
				[&](size_t order, khanNextPreferred _) { cutCount++; });
	}

	EXPECT_EQ(execOrder.size(), 4);
	EXPECT_EQ(cutCount, 2);
}

TEST(VVarDependencyGraphTest, scheduleTest)
{
	auto model = makeModel();
	VVarDependencyGraph graph(model);
	SccLookup sccContent(graph);
	auto scheduledModel = modelica::schedule(std::move(model));
	EXPECT_EQ(scheduledModel.getEquations().size(), 2);
	for (const auto& ass : scheduledModel.getEquations())
		EXPECT_EQ(ass.isForward(), true);
}
