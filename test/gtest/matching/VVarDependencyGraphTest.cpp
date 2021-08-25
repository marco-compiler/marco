#include "gtest/gtest.h"
#include <mlir/Support/LogicalResult.h>
#include <marco/mlirlowerer/passes/matching/Matching.h>
#include <marco/mlirlowerer/passes/matching/SCCCollapsing.h>
#include <marco/mlirlowerer/passes/matching/SVarDependencyGraph.h>
#include <marco/mlirlowerer/passes/matching/Schedule.h>
#include <marco/mlirlowerer/passes/matching/VVarDependencyGraph.h>
#include <marco/mlirlowerer/passes/model/Model.h>

#include "../TestingUtils.h"

using namespace marco::codegen::model;

std::string stringModel = "model Test "
													"int[2] x; "
													"int[2] y; "
													"equation "
													"for i in 1:2 loop "
													"x[i] = 3; "
													"end for; "
													"for i in 3:4 loop "
													"y[i-2] = x[i-2]; "
													"end for; "
													"end Test; ";

TEST(VVarDependencyGraphTest, CountTest)
{
	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	if (failed(match(model, 1000)))
		FAIL();

	VVarDependencyGraph graph(model);
	EXPECT_EQ(graph.count(), 2);
}

TEST(VVarDependencyGraphTest, PartitionTest)
{
	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	if (failed(match(model, 1000)))
		FAIL();

	VVarDependencyGraph graph(model);
	SCCLookup scc(graph);
	EXPECT_EQ(scc.count(), 2);
}

TEST(VVarDependencyGraphTest, GraphIteratorTest)
{
	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	if (failed(match(model, 1000)))
		FAIL();

	VVarDependencyGraph graph(model);
	SCCLookup sccContent(graph);

	int visitedNodes = 0;
	for (auto& scc : sccContent)
		for (auto& vertex : scc.range(graph))
			visitedNodes++;

	EXPECT_EQ(visitedNodes, graph.count());
}

TEST(VVarDependencyGraphTest, ScalarGraph)
{
	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	if (failed(match(model, 1000)))
		FAIL();

	VVarDependencyGraph graph(model);
	SCCLookup sccContent(graph);

	int visitedNodes = 0;
	for (const auto& scc : sccContent)
	{
		SVarDependencyGraph scalarGraph(graph, scc);
		visitedNodes += scalarGraph.count();
	}
	EXPECT_EQ(visitedNodes, 4);
}

TEST(VVarDependencyGraphTest, TestOrder)
{
	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	if (failed(match(model, 1000)))
		FAIL();

	VVarDependencyGraph graph(model);
	SCCLookup sccContent(graph);
	llvm::SmallVector<SVarDependencyGraph, 0> sccs;
	llvm::SmallVector<size_t, 0> execOrder;
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

TEST(VVarDependencyGraphTest, ScheduleTest)
{
	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	if (failed(match(model, 1000)))
		FAIL();

	if (failed(solveSCCs(model, 1000)))
		FAIL();

	if (failed(schedule(model)))
		FAIL();

	EXPECT_EQ(model.getEquations().size(), 2);
	for (const Equation& ass : model.getEquations())
	{
		EXPECT_EQ(ass.isForward(), true);
	}
}
