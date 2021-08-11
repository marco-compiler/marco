
#include "TestingUtils.h"

#include "gtest/gtest.h"
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>
#include <modelica/frontend/Parser.h>
#include <modelica/frontend/Passes.h>
#include <modelica/mlirlowerer/CodeGen.h>
#include <modelica/mlirlowerer/passes/model/Model.h>

void makeModel(
		mlir::MLIRContext& context,
		std::string& stringModel,
		modelica::codegen::model::Model& model)
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

	modelica::codegen::ModelicaBuilder builder(&context);

	modelica::codegen::MLIRLowerer lowerer(context);
	auto moduleOp = lowerer.run(classes);
	if (!moduleOp)
		FAIL();

	auto result = modelica::codegen::getUnmatchedModel(*moduleOp);
	if (!result)
		FAIL();

	model = *result;
}

void makeSolvedModel(
		mlir::MLIRContext& context,
		std::string& stringModel,
		modelica::codegen::model::Model& model)
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

	modelica::codegen::ModelicaBuilder builder(&context);

	modelica::codegen::MLIRLowerer lowerer(context);
	auto moduleOp = lowerer.run(classes);
	if (!moduleOp)
		FAIL();

	modelica::codegen::SolveModelOptions solveModelOptions;
	solveModelOptions.solver = modelica::codegen::CleverDAE;

	auto result = modelica::codegen::getSolvedModel(*moduleOp, solveModelOptions);
	if (!result)
		FAIL();

	model = *result;
}
