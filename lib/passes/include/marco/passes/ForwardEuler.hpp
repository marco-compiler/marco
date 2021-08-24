#pragma once
#include "marco/model/AssignModel.hpp"
#include "marco/model/Model.hpp"

namespace marco
{
	llvm::Expected<AssignModel> addApproximation(Model& model, double deltaTime);
}	 // namespace marco
