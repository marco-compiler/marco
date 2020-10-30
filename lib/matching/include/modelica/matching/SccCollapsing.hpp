#pragma once

#include <llvm/Support/Error.h>

#include "modelica/model/Model.hpp"

namespace modelica
{
	llvm::Expected<Model> solveScc(Model&& model, size_t maxIterations = 100);
}
