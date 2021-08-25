#pragma once
#include "gtest/gtest.h"
#include <mlir/IR/MLIRContext.h>
#include <marco/mlirlowerer/passes/model/Model.h>

void makeModel(
		mlir::MLIRContext& context,
		std::string& stringModel,
		marco::codegen::model::Model& model);

void makeSolvedModel(
		mlir::MLIRContext& context,
		std::string& stringModel,
		marco::codegen::model::Model& model);
