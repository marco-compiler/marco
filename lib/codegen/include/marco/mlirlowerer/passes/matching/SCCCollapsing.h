#pragma once

#include <llvm/Support/Error.h>
#include <mlir/IR/Builders.h>
#include <marco/mlirlowerer/passes/model/Model.h>

namespace marco::codegen::model
{
	mlir::LogicalResult solveSCCs(mlir::OpBuilder& builder, Model& model, size_t maxIterations);
}
