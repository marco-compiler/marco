#pragma once
#include "modelica/model/AssignModel.hpp"
#include "modelica/model/Model.hpp"

namespace modelica
{
	llvm::Expected<AssignModel> addApproximation(Model& model, double deltaTime);
}	 // namespace modelica
