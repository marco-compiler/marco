#pragma once
#include "gtest/gtest.h"
#include <mlir/Pass/PassManager.h>
#include <modelica/frontend/Parser.h>
#include <modelica/frontend/Passes.h>
#include <modelica/mlirlowerer/CodeGen.h>
#include <modelica/mlirlowerer/passes/model/Model.h>

static void makeModel(
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
	frontendPassManager.addPass(modelica::frontend::createBreakRemovingPass());
	frontendPassManager.addPass(modelica::frontend::createReturnRemovingPass());

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
