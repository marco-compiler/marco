#include "gtest/gtest.h"
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <modelica/frontend/Parser.h>
#include <modelica/frontend/Passes.h>
#include <modelica/mlirlowerer/CodeGen.h>
#include <modelica/mlirlowerer/passes/matching/Edge.h>
#include <modelica/mlirlowerer/passes/matching/Flow.h>
#include <modelica/mlirlowerer/passes/matching/Matching.h>
#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/VectorAccess.h>

using namespace modelica::codegen::model;

std::string stringModel1 = "model Test "
													 "int[10] x = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; "
													 "int[10] y = {3, 3, 3, 3, 3, 3, 3, 3, 3, 3}; "
													 "equation "
													 "for i in 1:5 loop "
													 "x[i] = y[i]; "
													 "end for; "
													 "for j in 1:5 loop "
													 "y[j] = x[j+2] + 2; "
													 "end for; "
													 "end Test; ";

std::string stringModel2 = "model Test "
													 "int[10] x = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; "
													 "equation "
													 "for i in 1:10 loop "
													 "x[i] = 0; "
													 "end for; "
													 "end Test; ";

TEST(MatchingTest, SimpleMatch)
{
	modelica::frontend::Parser parser(stringModel1);
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

	MatchingGraph graph(*model);

	EXPECT_EQ(graph.variableCount(), 2);
	EXPECT_EQ(graph.equationCount(), 2);
	EXPECT_EQ(graph.edgesCount(), 4);

	if (failed(graph.match(4)))
		FAIL();
	EXPECT_EQ(graph.matchedEdgesCount(), 2);
	EXPECT_EQ(graph.matchedCount(), 10);
}

TEST(MatchingTest, FirstMatchingSize)
{
	modelica::frontend::Parser parser(stringModel1);
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

	MatchingGraph graph(*model);
	AugmentingPath path(graph);
	FlowCandidates res = path.selectStartingEdge();
	EXPECT_EQ(res.getCurrent().getSet().size(), 5);
	EXPECT_EQ(res.getCurrent().getEdge().getSet(), modelica::IndexSet());
	EXPECT_EQ(res.getCurrent().isForwardEdge(), true);
}

TEST(MatchingTest, EmptyGraph)
{
	modelica::frontend::Parser parser(stringModel1);
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

	Model emptyModel = Model((*model).getOp(), {}, {});
	MatchingGraph graph(emptyModel);
	if (failed(graph.match(4)))
		FAIL();
	EXPECT_EQ(graph.matchedEdgesCount(), 0);
}

TEST(MatchingTest, TestMatchingFailure)
{
	modelica::frontend::Parser parser(stringModel1);
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

	EXPECT_TRUE(failed(match(*model, 1000)));
}

TEST(MatchingTest, UnsuccessfulMatchingTest)
{
	modelica::frontend::Parser parser(stringModel1);
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

	modelica::codegen::MLIRLowerer lowerer(
			context, modelica::codegen::ModelicaOptions::getDefaultOptions());
	auto moduleOp = lowerer.run(classes);
	if (!moduleOp)
		FAIL();

	auto model = modelica::codegen::getUnmatchedModel(*moduleOp);
	if (!model)
		FAIL();

	if (!failed(match(*model, 1000)))
		FAIL();
}

TEST(MatchingTest, SuccessfulMatchingTest)
{
	modelica::frontend::Parser parser(stringModel2);
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

	modelica::codegen::MLIRLowerer lowerer(
			context, modelica::codegen::ModelicaOptions::getDefaultOptions());
	auto moduleOp = lowerer.run(classes);
	if (!moduleOp)
		FAIL();

	auto model = modelica::codegen::getUnmatchedModel(*moduleOp);
	if (!model)
		FAIL();

	if (failed(match(*model, 1000)))
		FAIL();
}

TEST(MatchingTest, BaseGraphScalarDependencies)
{
	modelica::frontend::Parser parser(stringModel2);
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

	modelica::codegen::MLIRLowerer lowerer(
			context, modelica::codegen::ModelicaOptions::getDefaultOptions());
	auto moduleOp = lowerer.run(classes);
	if (!moduleOp)
		FAIL();

	auto model = modelica::codegen::getUnmatchedModel(*moduleOp);
	if (!model)
		FAIL();

	MatchingGraph graph(*model);
	auto range = graph.arcsOf(model->getEquations()[0]);
	EXPECT_EQ(graph.equationCount(), 1);
	EXPECT_EQ(graph.variableCount(), 1);
	EXPECT_EQ(std::distance(range.begin(), range.end()), 1);
}

TEST(MatchingTest, ScalarMatchingTest)
{
	modelica::frontend::Parser parser(stringModel2);
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

	modelica::codegen::MLIRLowerer lowerer(
			context, modelica::codegen::ModelicaOptions::getDefaultOptions());
	auto moduleOp = lowerer.run(classes);
	if (!moduleOp)
		FAIL();

	auto model = modelica::codegen::getUnmatchedModel(*moduleOp);
	if (!model)
		FAIL();

	MatchingGraph graph(*model);
	if (failed(graph.match(4)))
		FAIL();
	EXPECT_EQ(graph.matchedEdgesCount(), 1);
	EXPECT_EQ(graph.matchedCount(), 10);
}
