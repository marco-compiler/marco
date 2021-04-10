#pragma once
#include "modelica/model/ModErrors.hpp"
#include "modelica/model/Model.hpp"

namespace modelica
{
	llvm::Error solveDer(Model& model);
}	 // namespace modelica
