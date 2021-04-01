#pragma once

#include <llvm/Support/Error.h>
#include <modelica/mlirlowerer/passes/model/Model.h>

namespace modelica::codegen::model
{
	llvm::Expected<Model> solveScc(Model&& model, size_t maxIterations = 100);
}
