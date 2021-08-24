#pragma once

#include <llvm/Support/Error.h>

#include "marco/model/Model.hpp"

namespace marco
{
	llvm::Expected<Model> solveScc(Model&& model, size_t maxIterations = 100);
}
