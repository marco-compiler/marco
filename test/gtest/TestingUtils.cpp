
#include "TestingUtils.h"

#include "gtest/gtest.h"
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>
#include <marco/frontend/Parser.h>
#include <marco/frontend/Passes.h>
#include <marco/mlirlowerer/CodeGen.h>
#include <marco/mlirlowerer/passes/model/Model.h>

void makeModel(
		mlir::MLIRContext& context,
		std::string& stringModel,
		marco::codegen::model::Model& model)
{
	marco::frontend::Parser parser(stringModel);
	auto ast = parser.classDefinition();
	if (!ast)
		FAIL();

	llvm::SmallVector<std::unique_ptr<marco::frontend::Class>, 3> classes;
	classes.push_back(std::move(*ast));

	marco::frontend::PassManager frontendPassManager;
	frontendPassManager.addPass(marco::frontend::createTypeCheckingPass());
	frontendPassManager.addPass(marco::frontend::createConstantFolderPass());

	if (frontendPassManager.run(classes))
		FAIL();

	marco::codegen::modelica::ModelicaBuilder builder(&context);

	marco::codegen::MLIRLowerer lowerer(context);
	auto moduleOp = lowerer.run(classes);
	if (!moduleOp)
		FAIL();

	auto result = marco::codegen::getUnmatchedModel(*moduleOp);
	if (!result)
		FAIL();

	model = *result;
}

void makeSolvedModel(
		mlir::MLIRContext& context,
		std::string& stringModel,
		marco::codegen::model::Model& model)
{
	marco::frontend::Parser parser(stringModel);
	auto ast = parser.classDefinition();
	if (!ast)
		FAIL();

	llvm::SmallVector<std::unique_ptr<marco::frontend::Class>, 3> classes;
	classes.push_back(std::move(*ast));

	marco::frontend::PassManager frontendPassManager;
	frontendPassManager.addPass(marco::frontend::createTypeCheckingPass());
	frontendPassManager.addPass(marco::frontend::createConstantFolderPass());

	if (frontendPassManager.run(classes))
		FAIL();

	marco::codegen::modelica::ModelicaBuilder builder(&context);

	marco::codegen::MLIRLowerer lowerer(context);
	auto moduleOp = lowerer.run(classes);
	if (!moduleOp)
		FAIL();

	marco::codegen::SolveModelOptions solveModelOptions;
	solveModelOptions.solver = marco::codegen::CleverDAE;

	auto result = marco::codegen::getSolvedModel(*moduleOp, solveModelOptions);
	if (!result)
		FAIL();

	model = *result;
}
