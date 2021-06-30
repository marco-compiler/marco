#include "gtest/gtest.h"
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <modelica/frontend/Parser.h>
#include <modelica/frontend/Passes.h>
#include <modelica/mlirlowerer/CodeGen.h>
#include <modelica/mlirlowerer/passes/matching/Matching.h>
#include <modelica/mlirlowerer/passes/matching/SCCCollapsing.h>
#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/VectorAccess.h>
#include <modelica/utils/Interval.hpp>

using namespace modelica::codegen::model;

TEST(SCCCollapsingTest, EquationShouldBeNormalizable)
{
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

	EXPECT_EQ(
			model->getEquations()[1].getInductions(),
			modelica::MultiDimInterval({ { 3, 5 } }));

	for (Equation& eq : model->getEquations())
		if (failed(eq.explicitate()))
			FAIL();

	for (Equation& eq : model->getEquations())
		if (failed(eq.normalize()))
			FAIL();

	auto acc = AccessToVar::fromExp(model->getEquations()[0].lhs());
	EXPECT_TRUE(acc.getAccess().isIdentity());
}

TEST(SCCCollapsingTest, ThreeDepthNormalization)
{
	std::string stringModel = "model Test "
														"int[2, 3, 4] x; "
														"equation "
														"for i in 1:2 loop "
														"for j in 1:3 loop "
														"for k in 1:4 loop "
														"x[i, j, k] = 5; "
														"end for; "
														"end for; "
														"end for; "
														"end Test; ";

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

	EXPECT_EQ(
			model->getEquations()[0].getInductions(),
			modelica::MultiDimInterval({ { 1, 3 }, { 1, 4 }, { 1, 5 } }));

	for (Equation& eq : model->getEquations())
		if (failed(eq.explicitate()))
			FAIL();

	for (Equation& eq : model->getEquations())
		if (failed(eq.normalize()))
			FAIL();

	auto acc = AccessToVar::fromExp(model->getEquations()[0].lhs());
	EXPECT_TRUE(acc.getAccess().isIdentity());
}

TEST(SCCCollapsingTest, CyclesWithScalarsSolved)
{
	std::string stringModel = "model Loop2 "
														"int x; "
														"int y; "
														"int z; "
														"int w; "
														"int v; "
														"equation "
														"x + y = 9 - v; "
														"x - y = 3; "
														"z + w = 1 + v; "
														"z - w = -1; "
														"v = 4; "
														"end Loop2; ";

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

	EXPECT_EQ(model->getVariables().size(), 5);
	EXPECT_EQ(model->getEquations().size(), 5);
	EXPECT_EQ(model->getBltBlocks().size(), 0);

	if (failed(solveSCCs(*model, 1000)))
		FAIL();

	EXPECT_EQ(model->getVariables().size(), 5);
	EXPECT_EQ(model->getEquations().size(), 5);
	EXPECT_EQ(model->getBltBlocks().size(), 0);
	for (const BltBlock& bltBlock : model->getBltBlocks())
		EXPECT_EQ(bltBlock.getEquations().size(), 2);
}

TEST(SCCCollapsingTest, CyclesWithVectorsInBltBlock)
{
	std::string stringModel = "model Loop6 "
														"int[5] x; "
														"equation "
														"x[1] + x[2] = 2; "
														"x[1] - x[2] = 4; "
														"x[3] + x[4] = 1; "
														"x[3] - x[4] = -1; "
														"x[5] = x[4] + x[1]; "
														"end Loop6; ";

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

	EXPECT_EQ(model->getVariables().size(), 1);
	EXPECT_EQ(model->getEquations().size(), 5);
	EXPECT_EQ(model->getBltBlocks().size(), 0);

	if (failed(solveSCCs(*model, 1000)))
		FAIL();

	EXPECT_EQ(model->getVariables().size(), 1);
	EXPECT_EQ(model->getEquations().size(), 1);
	EXPECT_EQ(model->getBltBlocks().size(), 2);
	for (auto& bltBlock : model->getBltBlocks())
		EXPECT_EQ(bltBlock.getEquations().size(), 2);
}

TEST(SCCCollapsingTest, CyclesWithThreeEquationsSolved)
{
	std::string stringModel = "model Loop4 "
														"Real x; "
														"Real y; "
														"Real z; "
														"equation "
														"x + y = 1.0; "
														"x + z = 2.0; "
														"y + z = 3.0; "
														"end Loop4; ";

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

	EXPECT_EQ(model->getVariables().size(), 3);
	EXPECT_EQ(model->getEquations().size(), 3);
	EXPECT_EQ(model->getBltBlocks().size(), 0);

	if (failed(solveSCCs(*model, 1000)))
		FAIL();

	EXPECT_EQ(model->getVariables().size(), 3);
	EXPECT_EQ(model->getEquations().size(), 3);
	EXPECT_EQ(model->getBltBlocks().size(), 0);
}

TEST(SCCCollapsingTest, CyclesWithForEquationsInBltBlock)
{
	std::string stringModel = "model Loop12 "
														"Real[4] x; "
														"equation "
														"for i in 1:1 loop "
														"x[i] + x[i+1] = x[4]; "
														"end for; "
														"for i in 1:1 loop "
														"x[i] - x[i+1] = 3.0; "
														"end for; "
														"x[2] = x[3]; "
														"2.0 = x[4]; "
														"end Loop12; ";

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

	EXPECT_EQ(model->getVariables().size(), 1);
	EXPECT_EQ(model->getEquations().size(), 4);
	EXPECT_EQ(model->getBltBlocks().size(), 0);

	if (failed(solveSCCs(*model, 1000)))
		FAIL();

	EXPECT_EQ(model->getVariables().size(), 1);
	EXPECT_EQ(model->getEquations().size(), 2);
	EXPECT_EQ(model->getBltBlocks().size(), 1);
	EXPECT_EQ(model->getBltBlocks()[0].getEquations().size(), 2);
}
