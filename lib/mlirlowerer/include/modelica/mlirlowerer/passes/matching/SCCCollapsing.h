#pragma once

#include <llvm/Support/Error.h>
#include <mlir/IR/Builders.h>
#include <modelica/mlirlowerer/passes/model/Model.h>

namespace modelica::codegen::model
{
	mlir::LogicalResult solveSCCs(mlir::OpBuilder& builder, Model& model, size_t maxIterations);
}
