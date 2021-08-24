#pragma once
#include "marco/model/ModErrors.hpp"
#include "marco/model/Model.hpp"

namespace marco
{
	llvm::Error solveDer(Model& model);
}	 // namespace marco
