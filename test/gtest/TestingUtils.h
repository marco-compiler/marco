#pragma once
#include "gtest/gtest.h"
#include <mlir/IR/MLIRContext.h>
#include <modelica/mlirlowerer/passes/model/Model.h>

void makeModel(
		mlir::MLIRContext& context,
		std::string& stringModel,
		modelica::codegen::model::Model& model);

void makeSolvedModel(
		mlir::MLIRContext& context,
		std::string& stringModel,
		modelica::codegen::model::Model& model);
