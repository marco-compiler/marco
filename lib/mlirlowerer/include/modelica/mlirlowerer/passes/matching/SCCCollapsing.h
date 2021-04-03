#pragma once

#include <llvm/Support/Error.h>
#include <modelica/mlirlowerer/passes/model/Model.h>

namespace modelica::codegen::model
{
	mlir::LogicalResult solveSCC(Model& model, size_t maxIterations);
}
