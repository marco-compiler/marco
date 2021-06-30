#include "gtest/gtest.h"
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <modelica/frontend/Parser.h>
#include <modelica/frontend/Passes.h>
#include <modelica/mlirlowerer/CodeGen.h>
#include <modelica/mlirlowerer/passes/matching/Matching.h>
#include <modelica/mlirlowerer/passes/matching/SCCCollapsing.h>
#include <modelica/mlirlowerer/passes/matching/SVarDependencyGraph.h>
#include <modelica/mlirlowerer/passes/matching/Schedule.h>
#include <modelica/mlirlowerer/passes/matching/VVarDependencyGraph.h>
#include <modelica/mlirlowerer/passes/model/Model.h>

using namespace modelica::codegen::model;

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
	modelica::frontend::Parser parser(stringModel);
	auto ast = parser.classDefinition();
	if (!ast)
		FAIL();

	llvm::SmallVector<std::unique_ptr<modelica::frontend::Class>, 3> classes;
	classes.push_back(std::move(*ast));

	modelica::frontend::PassManager frontendPassManager;
	frontendPassManager.addPass(modelica::frontend::createTypeCheckingPass());
	frontendPassManager.addPass(modelica::frontend::createConstantFolderPass());

	if (frontendPassManager.run(classes))
		FAIL();

	mlir::MLIRContext context;
	modelica::codegen::ModelicaBuilder builder(&context);

	modelica::codegen::MLIRLowerer lowerer(context);
	auto moduleOp = lowerer.run(classes);
	if (!moduleOp)
		FAIL();

	auto model = modelica::codegen::getUnmatchedModel(*moduleOp);
	if (!model)
		FAIL();

	if (failed(match(*model, 1000)))
		FAIL();

	VVarDependencyGraph graph(*model);
	EXPECT_EQ(graph.count(), 2);
}

TEST(VVarDependencyGraphTest, PartitionTest)
{
	modelica::frontend::Parser parser(stringModel);
	auto ast = parser.classDefinition();
	if (!ast)
		FAIL();

	llvm::SmallVector<std::unique_ptr<modelica::frontend::Class>, 3> classes;
	classes.push_back(std::move(*ast));

	modelica::frontend::PassManager frontendPassManager;
	frontendPassManager.addPass(modelica::frontend::createTypeCheckingPass());
	frontendPassManager.addPass(modelica::frontend::createConstantFolderPass());

	if (frontendPassManager.run(classes))
		FAIL();

	mlir::MLIRContext context;
	modelica::codegen::ModelicaBuilder builder(&context);

	modelica::codegen::MLIRLowerer lowerer(context);
	auto moduleOp = lowerer.run(classes);
	if (!moduleOp)
		FAIL();

	auto model = modelica::codegen::getUnmatchedModel(*moduleOp);
	if (!model)
		FAIL();

	if (failed(match(*model, 1000)))
		FAIL();

	VVarDependencyGraph graph(*model);
	SCCLookup scc(graph);
	EXPECT_EQ(scc.count(), 2);
}

TEST(VVarDependencyGraphTest, GraphIteratorTest)
{
	modelica::frontend::Parser parser(stringModel);
	auto ast = parser.classDefinition();
	if (!ast)
		FAIL();

	llvm::SmallVector<std::unique_ptr<modelica::frontend::Class>, 3> classes;
	classes.push_back(std::move(*ast));

	modelica::frontend::PassManager frontendPassManager;
	frontendPassManager.addPass(modelica::frontend::createTypeCheckingPass());
	frontendPassManager.addPass(modelica::frontend::createConstantFolderPass());

	if (frontendPassManager.run(classes))
		FAIL();

	mlir::MLIRContext context;
	modelica::codegen::ModelicaBuilder builder(&context);

	modelica::codegen::MLIRLowerer lowerer(context);
	auto moduleOp = lowerer.run(classes);
	if (!moduleOp)
		FAIL();

	auto model = modelica::codegen::getUnmatchedModel(*moduleOp);
	if (!model)
		FAIL();

	if (failed(match(*model, 1000)))
		FAIL();

	VVarDependencyGraph graph(*model);
	SCCLookup sccContent(graph);

	int visitedNodes = 0;
	for (auto& scc : sccContent)
		for (auto& vertex : scc.range(graph))
			visitedNodes++;

	EXPECT_EQ(visitedNodes, graph.count());
}

TEST(VVarDependencyGraphTest, ScalarGraph)
{
	modelica::frontend::Parser parser(stringModel);
	auto ast = parser.classDefinition();
	if (!ast)
		FAIL();

	llvm::SmallVector<std::unique_ptr<modelica::frontend::Class>, 3> classes;
	classes.push_back(std::move(*ast));

	modelica::frontend::PassManager frontendPassManager;
	frontendPassManager.addPass(modelica::frontend::createTypeCheckingPass());
	frontendPassManager.addPass(modelica::frontend::createConstantFolderPass());

	if (frontendPassManager.run(classes))
		FAIL();

	mlir::MLIRContext context;
	modelica::codegen::ModelicaBuilder builder(&context);

	modelica::codegen::MLIRLowerer lowerer(context);
	auto moduleOp = lowerer.run(classes);
	if (!moduleOp)
		FAIL();

	auto model = modelica::codegen::getUnmatchedModel(*moduleOp);
	if (!model)
		FAIL();

	if (failed(match(*model, 1000)))
		FAIL();

	VVarDependencyGraph graph(*model);
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
	modelica::frontend::Parser parser(stringModel);
	auto ast = parser.classDefinition();
	if (!ast)
		FAIL();

	llvm::SmallVector<std::unique_ptr<modelica::frontend::Class>, 3> classes;
	classes.push_back(std::move(*ast));

	modelica::frontend::PassManager frontendPassManager;
	frontendPassManager.addPass(modelica::frontend::createTypeCheckingPass());
	frontendPassManager.addPass(modelica::frontend::createConstantFolderPass());

	if (frontendPassManager.run(classes))
		FAIL();

	mlir::MLIRContext context;
	modelica::codegen::ModelicaBuilder builder(&context);

	modelica::codegen::MLIRLowerer lowerer(context);
	auto moduleOp = lowerer.run(classes);
	if (!moduleOp)
		FAIL();

	auto model = modelica::codegen::getUnmatchedModel(*moduleOp);
	if (!model)
		FAIL();

	if (failed(match(*model, 1000)))
		FAIL();

	VVarDependencyGraph graph(*model);
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
	modelica::frontend::Parser parser(stringModel);
	auto ast = parser.classDefinition();
	if (!ast)
		FAIL();

	llvm::SmallVector<std::unique_ptr<modelica::frontend::Class>, 3> classes;
	classes.push_back(std::move(*ast));

	modelica::frontend::PassManager frontendPassManager;
	frontendPassManager.addPass(modelica::frontend::createTypeCheckingPass());
	frontendPassManager.addPass(modelica::frontend::createConstantFolderPass());

	if (frontendPassManager.run(classes))
		FAIL();

	mlir::MLIRContext context;
	modelica::codegen::ModelicaBuilder builder(&context);

	modelica::codegen::MLIRLowerer lowerer(context);
	auto moduleOp = lowerer.run(classes);
	if (!moduleOp)
		FAIL();

	auto model = modelica::codegen::getUnmatchedModel(*moduleOp);
	if (!model)
		FAIL();

	if (failed(match(*model, 1000)))
		FAIL();

	if (failed(solveSCCs(*model, 1000)))
		FAIL();

	if (failed(schedule(*model)))
		FAIL();

	EXPECT_EQ(model->getEquations().size(), 2);
	for (const Equation& ass : model->getEquations())
	{
		EXPECT_EQ(ass.isForward(), true);
	}
}
