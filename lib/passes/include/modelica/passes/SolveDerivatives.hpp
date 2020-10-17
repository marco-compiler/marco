#pragma once
#include "modelica/model/AssignModel.hpp"
#include "modelica/model/ModErrors.hpp"
#include "modelica/model/Model.hpp"

namespace modelica
{
	llvm::Expected<AssignModel> addAproximation(Model& model, double deltaTime);

	llvm::Error solveDer(Model& model);
}	 // namespace modelica
